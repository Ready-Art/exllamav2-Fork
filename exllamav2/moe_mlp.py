from __future__ import annotations
import os
import torch
import torch.nn.functional as F
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.layernorm import ExLlamaV2LayerNorm
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.lora import ExLlamaV2Lora
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.tensor_p import BROADCAST_ID, BROADCAST_RS, BROADCAST_VC

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

class ExLlamaV2MoEMLP(ExLlamaV2Module):

    name: str = "MoE MLP"

    layer_idx: int
    post_attention_layernorm: ExLlamaV2RMSNorm | ExLlamaV2LayerNorm
    w1: list
    w2: list
    w3: list
    gate: ExLlamaV2Linear
    num_experts: int
    num_experts_per_token: int

    q_handle: int | None

    temp_lora_size: int

    def __init__(
        self,
        model: ExLlamaV2,
        key: str,
        layer_idx: int,
        archparams = None
    ):
        super().__init__(model, key, archparams)

        cfg = self.model.config
        ap = self.archparams
        km = self.archparams.keys

        self.layer_idx = layer_idx

        self.q_handle = None
        self.temp_lora_size = 0

        hidden_size = cfg.hidden_size
        intermediate_size = cfg.intermediate_size
        self.num_experts = cfg.num_experts
        self.num_experts_per_token = cfg.num_experts_per_token

        # CPU/RAM expert offload (opt-in EXL2_MOE_OFFLOAD / tabby model.moe_offload_experts):
        # the LAST N experts of every MoE layer live in pinned host RAM; one shared GPU
        # staging slot per layer is H2D-filled on demand when routing selects them.
        # N >= num_experts offloads ALL experts. Forces the torch expert loop (the fused
        # q_moe_mlp kernel needs every expert resident). Single-device load path only.
        self.offload_experts = max(0, min(int(os.environ.get("EXL2_MOE_OFFLOAD", "0") or 0),
                                          self.num_experts))

        # MP expert layout: "shard" splits each expert's FFN across ranks (column-parallel
        # w1/w3, row-parallel w2) like a dense MLP — ~world× less expert VRAM/RAM per rank
        # and ~world× less offload PCIe traffic. "replicate" keeps the full MoE on every
        # rank (graph-safe fused kernel). EXL2_MOE_TP_SHARD=1 opts in; default 0 = replicate.
        # EXPERIMENTAL: sharding runs on 2/3/4 GPUs but Qwen3-family MP attention has an
        # open parity bug (see README), so sharded output is not yet verified coherent.
        self._tp_shard = os.environ.get("EXL2_MOE_TP_SHARD", "0") != "0"

        if ap.norm == "layernorm":
            self.post_attention_layernorm = ExLlamaV2LayerNorm(model, key + km["norm_2"])
        elif ap.norm == "rmsnorm":
            self.post_attention_layernorm = ExLlamaV2RMSNorm(model, key + km["norm_2"])

        w1_key = key + km["mlp_gate"]
        w2_key = key + km["mlp_down"]
        w3_key = key + km["mlp_up"]
        w1_f_key = w1_key.replace(".*.", ".")
        w2_f_key = w2_key.replace(".*.", ".")
        w3_f_key = w3_key.replace(".*.", ".")

        gate_key = km["mlp_expert_gate"]

        self.w1 = []
        self.w2 = []
        self.w3 = []

        bu = 0
        # bd = 0
        for e in range(self.num_experts):
            au = bu
            # ad = bd
            bu += intermediate_size
            # bd += hidden_size
            w1 = ExLlamaV2Linear(model, w1_key.replace("*", str(e)), hidden_size, intermediate_size, ap.mlp_bias, f_key = w1_f_key, f_beg = au, f_end = bu)
            w2 = ExLlamaV2Linear(model, w2_key.replace("*", str(e)), intermediate_size, hidden_size, ap.mlp_bias, f_key = w2_f_key, f_beg = au, f_end = bu)
            w3 = ExLlamaV2Linear(model, w3_key.replace("*", str(e)), hidden_size, intermediate_size, ap.mlp_bias, f_key = w3_f_key, f_beg = au, f_end = bu)
            self.w1.append(w1)
            self.w2.append(w2)
            self.w3.append(w3)

        self.gate = ExLlamaV2Linear(model, key + gate_key, hidden_size, self.num_experts, False, pad32 = False)

        self.submodules = [self.post_attention_layernorm,
                           self.gate] + \
                           self.w1 + \
                           self.w2 + \
                           self.w3

        self.is_tp = False
        self.q_handle = None


    def numel(self) -> int:

        return sum(l.numel() for l in self.w1 + self.w2 + self.w3)


    @torch.inference_mode
    def load(self):

        self.post_attention_layernorm.load()
        self.gate.load()
        off = self.offload_experts
        first_off = self.num_experts - off
        # One shared staging slot per projection across the offloaded experts (all
        # same-shape): GPU holds ONE expert's data, not N. _build_offload fills the
        # shared dict from the first offloaded expert and reuses it for the rest.
        shared = [{}, {}, {}] if off else None
        for e in range(self.num_experts):
            is_off = off and e >= first_off
            for wi, wl in enumerate((self.w1, self.w2, self.w3)):
                wl[e].offload = bool(is_off)
                if is_off:
                    wl[e]._offload_shared = shared[wi]
            self.w1[e].load()
            self.w2[e].load()
            self.w3[e].load()
        if off:
            print(f"[moe-offload] layer {self.layer_idx}: {off}/{self.num_experts} experts "
                  f"in pinned RAM (shared GPU staging slot)", flush = True)

        if self.w1[0].is_quant() and not off:
            device_context = self.model.get_device_context(self.device_idx)
            device_context.begin_scratch_alloc()
            self.q_handle = ext_c.make_q_moe_mlp(
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.bias if self.post_attention_layernorm.bias is not None else none_tensor,
                isinstance(self.post_attention_layernorm, ExLlamaV2RMSNorm),
                self.post_attention_layernorm.variance_epsilon,
                self.gate.linear.weight,
                self.num_experts,
                self.num_experts_per_token,
                [w.q_handle for w in self.w1],
                [w.q_handle for w in self.w2],
                [w.q_handle for w in self.w3],
                device_context.get_scratch_slice(self.temp_state_size()),
                device_context.get_scratch_slice(self.temp_gathered_state_size()),
                device_context.get_scratch_slice(self.temp_a_size()),
                device_context.get_scratch_slice(self.temp_b_size()),
                device_context.get_scratch_slice(self.temp_logit_size()),
                device_context.get_scratch_slice(self.temp_dq_size()),
                self.model.config.max_input_len * self.model.config.max_batch_size,
                self.archparams.mlp_act_func == "gelu"
            )


    def unload(self):
        if self.q_handle is not None:
            ext_c.free_q_moe_mlp(self.q_handle)
            self.q_handle = None

        self.post_attention_layernorm.unload()
        self.gate.unload()
        for e in range(self.num_experts):
            self.w1[e].unload()
            self.w2[e].unload()
            self.w3[e].unload()


    def weight_footprint(self) -> int:

        # Offloaded experts: GPU holds one shared staging slot per projection (the
        # first offloaded expert's footprint) + small resident constants; the rest
        # lives in pinned host RAM.
        off = self.offload_experts
        n_gpu = self.num_experts - off
        fp = self.post_attention_layernorm.weight_footprint() + self.gate.weight_footprint()
        for wl in (self.w1, self.w2, self.w3):
            fp += sum(wl[e].weight_footprint() for e in range(n_gpu))
            if off:
                fp += wl[n_gpu].weight_footprint()   # shared staging slot
        return fp


    def scratch_space_fixed(self) -> int:

        return self.temp_state_size() + \
               self.temp_gathered_state_size() + \
               self.temp_a_size() + \
               self.temp_b_size() + \
               self.temp_logit_size() + \
               self.temp_dq_size()


    def scratch_space(self) -> int:

        # assert self.model.config.intermediate_size >= self.model.config.hidden_size
        return self.temp_state_size() + \
               self.temp_gathered_state_size() + \
               self.temp_a_size() + \
               self.temp_b_size() + \
               self.temp_logit_size() + \
               self.temp_dq_size()


    def temp_state_size(self) -> int:

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.hidden_size * 2 + 128


    def temp_gathered_state_size(self) -> int:

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.hidden_size * 2 + 128


    def temp_a_size(self) -> int:

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.intermediate_size * 2 + 128


    def temp_b_size(self) -> int:

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.intermediate_size * 2 + 128


    def temp_dq_size(self) -> int:

        return max(self.w1[0].temp_dq_size(),
                   self.w2[0].temp_dq_size(),
                   self.w3[0].temp_dq_size())


    def temp_logit_size(self) -> int:

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.num_experts * 2 + 128


    def set_device_idx(self, idx: int | None):
        super().set_device_idx(idx)

        self.post_attention_layernorm.set_device_idx(idx)
        self.gate.set_device_idx(idx)
        for e in range(self.num_experts):
            self.w1[e].set_device_idx(idx)
            self.w2[e].set_device_idx(idx)
            self.w3[e].set_device_idx(idx)


    def scratch_space_tp(self):
        devs = self.model.tp_context.num_devices
        if not self._tp_shard:
            # Replicated MoE: every rank runs the full MoE; reserve single-device scratch.
            return [self.scratch_space()] * devs
        # Sharded MoE mirrors the dense MLP: column-parallel w1/w3 (ID split),
        # column-parallel w2 on the RS split. Expert temp_dq slices reuse the device
        # scratch (each tp_load_rank calls begin_scratch_alloc), so one max suffices.
        cfg = self.model.config
        ctx = self.model.tp_context
        scratch = [0] * devs
        def amax(res):
            for i, s in enumerate(res): scratch[i] = max(scratch[i], s)
        amax(self.w1[0].scratch_space_tp(BROADCAST_ID, 1))
        amax(self.w3[0].scratch_space_tp(BROADCAST_ID, 1))
        amax(self.w2[0].scratch_space_tp(BROADCAST_RS, 1))
        maxrows = cfg.max_batch_size * cfg.max_input_len
        def add(res):
            for i, s in enumerate(res): scratch[i] += s
        add(ctx.get_temp_tensors_bc_s(maxrows, 2, BROADCAST_ID))
        add(ctx.get_temp_tensors_s(maxrows, 2, BROADCAST_RS))
        return scratch

    def tp_load_rank(self, rank: int):
        self.device_idx = rank
        off = self.offload_experts
        first_off = self.num_experts - off

        if self._tp_shard:
            # Sharded-expert MP MoE: each expert's FFN is split across ranks exactly like
            # a dense MLP (w1/w3 column-parallel ID, w2 column-parallel RS + unmap remap).
            # The router gate + input norm stay replicated so routing is identical on every
            # rank; forward_tp_sharded allgathers the summed expert output once per block.
            # Offload moves each rank's SHARD of the last N experts to pinned RAM (~world×
            # less RAM/PCIe than replicated offload).
            self.post_attention_layernorm.tp_load_rank(rank, BROADCAST_RS)
            self.gate.set_device_idx(rank)
            self.gate.load()
            shared = [{}, {}, {}] if off else None
            for e in range(self.num_experts):
                is_off = bool(off and e >= first_off)
                for wi, wl in enumerate((self.w1, self.w2, self.w3)):
                    wl[e].offload = is_off
                    if is_off:
                        wl[e]._offload_shared = shared[wi]
                perm = self.w2[e].tp_load_rank(rank, BROADCAST_RS, unmap = True)
                self.w1[e].tp_load_rank(rank, BROADCAST_ID, output_map = perm)
                self.w3[e].tp_load_rank(rank, BROADCAST_ID, output_map = perm)
            if off:
                print(f"[moe-offload] rank {rank} layer {self.layer_idx}: {off}/{self.num_experts} "
                      f"expert SHARDS in pinned RAM (sharded MP MoE)", flush = True)
            self.is_tp = True
            self.device_idx = rank
            return

        # Replicated-expert MP MoE: each rank loads the FULL gate/norm/experts and runs
        # them identically (forward_tp_old gathers/shards the residual around the block).
        # Resident experts use the graph-safe fused kernel; offload moves the last N
        # experts to pinned RAM on EVERY rank with one shared GPU staging slot/projection.
        self.post_attention_layernorm.set_device_idx(rank)
        self.gate.set_device_idx(rank)
        shared = [{}, {}, {}] if off else None
        for e in range(self.num_experts):
            is_off = bool(off and e >= first_off)
            for wi, wl in enumerate((self.w1, self.w2, self.w3)):
                wl[e].offload = is_off
                wl[e].set_device_idx(rank)
                if is_off:
                    wl[e]._offload_shared = shared[wi]
        self.post_attention_layernorm.load()
        self.gate.load()
        for e in range(self.num_experts):
            self.w1[e].load()
            self.w2[e].load()
            self.w3[e].load()
        if off:
            print(f"[moe-offload] rank {rank} layer {self.layer_idx}: {off}/{self.num_experts} "
                  f"experts in pinned RAM (replicated MP MoE)", flush = True)
        self.is_tp = True
        self.device_idx = rank

    def _forward_tp_sharded(self, full, ctx, loras):
        # Routing is replicated (gate + norm full on every rank) so selected_experts and
        # routing_weights agree across ranks. Each rank computes its FFN shard of every
        # selected expert (w1/w3 ID-shard -> act -> w2 RS-shard), accumulates the weighted
        # expert outputs into a local RS-shard of the hidden, then ONE allgather rebuilds
        # the full MoE output. Mirrors the dense-MLP column->RS->gather flow per expert.
        B, q, H = full.shape
        x = full.view(-1, H)
        post_norm = self.post_attention_layernorm.forward(x)
        router_logits = self.gate.forward(post_norm, loras = loras)
        routing_weights = F.softmax(router_logits, dim = -1, dtype = torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_token, dim = -1)
        routing_weights /= routing_weights.sum(dim = -1, keepdim = True)
        routing_weights = routing_weights.to(post_norm.dtype)

        dev, a, b = ctx.get_split(BROADCAST_RS)[0]
        hidden_local = b - a
        rows = x.shape[0]
        final_shard = torch.zeros((rows, hidden_local), dtype = post_norm.dtype, device = x.device)
        expert_mask = F.one_hot(selected_experts, num_classes = self.num_experts).permute(2, 1, 0)
        act_silu = self.archparams.mlp_act_func == "silu"
        first_off = self.num_experts - self.offload_experts

        for e in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[e])
            if top_x.shape[0] == 0:
                continue
            if self.offload_experts and e >= first_off:
                self.w1[e]._stage_offload()
                self.w3[e]._stage_offload()
                self.w2[e]._stage_offload()
            cur = post_norm[top_x]
            # forward_tp expects a per-device LIST (the TP GEMM binding takes Sequence[Tensor]);
            # under rank-local MP the list has one entry (this rank's shard).
            g = self.w1[e].forward_tp([cur], output_split = True, skip_broadcast = True, loras = loras)[0]
            u = self.w3[e].forward_tp([cur], output_split = True, skip_broadcast = True, loras = loras)[0]
            act = F.silu(g) if act_silu else F.gelu(g, approximate = "tanh")
            act = act * u
            d = self.w2[e].forward_tp([act], output_split = True, skip_broadcast = True, loras = loras)[0]
            d = d * routing_weights[top_x, idx, None]
            final_shard.index_add_(0, top_x, d.to(final_shard.dtype))

        final_full = ctx.allgather(1, [final_shard], BROADCAST_RS, BROADCAST_RS)[0]
        return final_full.view(B, q, H)

    def forward_tp_old(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:
        ctx = self.model.tp_context
        batch_size, q_len, _ = hidden_states.shape
        repl_chunk = ctx.decode_repl and q_len <= 32
        if ctx.seqpar and not repl_chunk:
            full = ctx.gather_residual(hidden_states)
        else:
            full = hidden_states

        if self._tp_shard:
            out_full = self._forward_tp_sharded(full, ctx, loras)
            if ctx.seqpar and not repl_chunk:
                return ctx.shard_residual(out_full) + hidden_states
            return out_full + hidden_states

        # Resident experts: the fused q_moe_mlp kernel handles routing internally with
        # fixed buffers -> CUDA-graph safe (TP graphs stay ON). Offloaded experts use the
        # data-dependent torch loop + H2D staging, which cannot be baked into a replayed
        # graph; that case runs eager (the graph manager disables capture when offload is
        # active). Replicated full-width MoE runs identically on every rank.
        rows = batch_size * q_len
        if (self.q_handle is not None and not intermediates and rows <= 4
                and self.num_experts in [4, 8, 16, 128]
                and not (loras is not None and len(loras) > 0)):
            ext_c.q_moe_mlp_forward_(self.q_handle, full.view(-1, full.shape[-1]))
            out = full
        else:
            out = self.forward_torch(full, cache, attn_params, past_len, intermediates,
                                     loras = loras, **kwargs)
        if ctx.seqpar and not repl_chunk:
            out = ctx.shard_residual(out)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        if self.is_tp:
            return self.forward_tp_old(hidden_states, cache, attn_params, past_len,
                                       intermediates, loras = loras, **kwargs)

        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # TODO: LoRA currently uses the Torch codepath. Needs conditional (early-exit) kernels with output scaling
        # for the LoRA matmuls in order to work with the C++ path

        if self.q_handle is None or intermediates or batch_size * sequence_length > 4 or self.num_experts not in [4, 8, 16, 128] or (loras is not None and len(loras) > 0):
            return self.forward_torch(hidden_states, cache, attn_params, past_len, intermediates, loras = loras, **kwargs)

        # if loras is None or self.temp_lora_size == 0:
        #     pass_loras = []
        #     pass_lora_temp = none_tensor
        # else:
        #     pass_loras = [id(x) for x in loras]
        #     pass_lora_temp = torch.empty((self.temp_lora_size,), dtype = torch.half, device = hidden_states.device)

        # ref = self.forward_torch(hidden_states, cache, attn_params, intermediates, loras = loras)
        # ext_c.q_moe_mlp_forward_(self.q_handle, hidden_states.view(-1, hidden_states.shape[-1]), pass_loras, pass_lora_temp)
        ext_c.q_moe_mlp_forward_(self.q_handle, hidden_states.view(-1, hidden_states.shape[-1]))

        return hidden_states


    def forward_torch(
        self,
        hidden_states: torch.Tensor,
        cache = None,
        attn_params = None,
        past_len = None,
        intermediates = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        residual = hidden_states

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Layernorm

        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        if intermediates: result = { "post_norm": hidden_states }

        # Get router logits

        router_logits = self.gate.forward(hidden_states, loras = loras)  #[:, :self.num_experts]

        # Get routing weights and select top K experts

        routing_weights = F.softmax(router_logits, dim = -1, dtype = torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_token, dim = -1)
        routing_weights /= routing_weights.sum(dim = -1, keepdim = True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype = hidden_states.dtype, device = hidden_states.device)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0: continue  # Skip experts that weren't selected at all

            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)

            if self.offload_experts and expert_idx >= self.num_experts - self.offload_experts:
                self.w1[expert_idx]._stage_offload()
                self.w3[expert_idx]._stage_offload()
                self.w2[expert_idx]._stage_offload()

            gate = self.w1[expert_idx].forward(current_state, loras = loras)
            up = self.w3[expert_idx].forward(current_state, loras = loras)

            if self.archparams.mlp_act_func == "silu":
                current_hidden_states = F.silu(gate)
            elif self.archparams.mlp_act_func == "gelu":
                current_hidden_states = F.gelu(gate)
            current_hidden_states *= up
            if intermediates: result[f"pre_down.{expert_idx}"] = current_hidden_states

            current_hidden_states = self.w2[expert_idx].forward(current_hidden_states, loras = loras)
            current_hidden_states *= routing_weights[top_x_list, idx_list, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        final_hidden_states += residual

        if intermediates:
            result["hidden_states"] = final_hidden_states
            return result
        else:
            return final_hidden_states


    def update_loras(self):
        pass
        # if self.q_handle is None: return
        #
        # w1_lora_a = []
        # w1_lora_b = []
        # w2_lora_a = []
        # w2_lora_b = []
        # w3_lora_a = []
        # w3_lora_b = []
        # for i in range(self.model.config.num_experts):
        #     w1_lora_a.append({ id(k): v for k, v in self.w1[i].lora_a_tensors.items() })
        #     w1_lora_b.append({ id(k): v for k, v in self.w1[i].lora_b_tensors.items() })
        #     w2_lora_a.append({ id(k): v for k, v in self.w2[i].lora_a_tensors.items() })
        #     w2_lora_b.append({ id(k): v for k, v in self.w2[i].lora_b_tensors.items() })
        #     w3_lora_a.append({ id(k): v for k, v in self.w3[i].lora_a_tensors.items() })
        #     w3_lora_b.append({ id(k): v for k, v in self.w3[i].lora_b_tensors.items() })
        #
        # temp_lora_size = ext_c.q_moe_mlp_set_loras(self.q_handle,
        #                                            w1_lora_a,
        #                                            w1_lora_b,
        #                                            w2_lora_a,
        #                                            w2_lora_b,
        #                                            w3_lora_a,
        #                                            w3_lora_b)
        #
        # self.temp_lora_size = temp_lora_size * self.model.config.max_batch_size * self.model.config.max_input_len


    def is_quant(self):
        return self.q_handle is not None


    def rank_reduce(self, k):

        for e in range(self.num_experts):
            self.w1[e].rank_reduce(k)
            self.w2[e].rank_reduce(k)
            self.w3[e].rank_reduce(k)
