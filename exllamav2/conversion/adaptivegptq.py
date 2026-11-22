from __future__ import annotations

import gc
import torch
from torch import nn
import torch.nn.functional as F
import math
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.util import list_live_tensors
from exllamav2.conversion.stager import WeightStager
import sys

class AdaptiveQuantizer:

    norm: float = 3.5
    max_p: float = 1.2
    min_p: float = 0.70
    p_grid: int = 96

    bits: int
    scale_bits: int
    scale_range: float = 1.0

    scale: torch.tensor
    qscale: torch.tensor
    qscale_max: float

    maxq: float
    scale_maxq: float
    qzero: float

    def __init__(self,
                 bits: int = 4,
                 scale_bits: int = 4):

        self.bits = bits
        self.scale_bits = scale_bits
        self.maxq = 2 ** bits - 1
        self.qzero = (self.maxq + 1) / 2
        self.scale_maxq = 2 ** scale_bits - 1

        self.scale_maxq = (2 ** self.scale_bits) - 1


    def find_params(self, x):

        xmax, _ = torch.max(torch.abs(x), dim = 0)
        xmax += 1e-12

        base_scale = xmax / (self.maxq / 2)
        qscale_max_t = torch.max(base_scale) * self.scale_range

        scale_tp = base_scale / qscale_max_t
        scale_tp = torch.sqrt(scale_tp)
        scale_tp *= (self.scale_maxq + 1)
        qscale_t = torch.clamp(torch.round(scale_tp), 1, self.scale_maxq + 1)
        qscale_tw = qscale_t / (self.scale_maxq + 1)
        qscale_tw = qscale_tw ** 2
        qscale_tw *= qscale_max_t

        q = torch.zeros((self.p_grid + 1, 128), dtype = torch.float, device = x.device)
        ext_c.quantize_err(x, q, qscale_tw, self.qzero, self.maxq, self.norm, self.min_p, self.max_p, self.p_grid)

        q = torch.sum(q, dim = 1)
        best_pi = torch.argmin(q)
        best_pif = best_pi / self.p_grid
        best_p = self.max_p * best_pif + self.min_p * (1 - best_pif)

        self.qscale = qscale_t.to(torch.short)
        self.scale = qscale_tw * best_p
        self.qscale_max = qscale_max_t * best_p

        # Make sure scales are rounded correctly for sanity test
        prescale = torch.tensor([1 / 256], dtype = torch.half, device = self.scale.device)
        self.scale = ((self.qscale * self.qscale).to(torch.half) * (self.qscale_max.half() * prescale)).float()


class AdaptiveGPTQ:

    percdamp: float = 0.12

    layer: nn.Linear
    device: torch.device

    group_size: int | dict
    bits: list
    bits_groups: list
    scale_bits: int
    hot_bits: int

    columns: int
    rows: int
    hessian: torch.tensor
    total_groups: int

    perm: torch.Tensor | None
    perm_cpu: torch.Tensor | None
    invperm: torch.Tensor | None

    # g_idx: torch.tensor = None
    scale: torch.Tensor | None
    qscale: torch.Tensor | None
    qscale_max: torch.Tensor | None
    qweight: torch.Tensor | None
    qgroups: torch.Tensor | None

    quant: torch.Tensor | None
    weights: torch.Tensor | None
    hessian: torch.Tensor | None
    hessian_inv: torch.Tensor | None
    num_samples: int = 0
    num_batches: int = 0

    quant_device: int = 0
    hessian_device: int = 0

    # Device the packed/scaled outputs are gathered onto. Defaults to 0 to preserve historical behaviour,
    # but the multi-GPU scheduler sets it to the worker's device so a quantized tensor is never bounced
    # across the bus just to be written to disk. quant-plan.md §6.4 (M2).
    output_device: int = 0

    # Device the *dequantized* weight is written back onto. This is NOT the worker device: multi-GPU quantization
    # spreads the work but leaves the model's device map untouched, so the reconstructed weight must land wherever
    # the owning module lives or the post-quantization forward pass in quantize.py() will matmul a cuda:N weight
    # against a cuda:M activation. ``None`` falls back to ``output_device`` (single-GPU behaviour).
    target_device: int | None = None

    # Optional WeightStager for pinned, overlapped weight upload. quant-plan.md §6.8 (M5).
    stager: WeightStager | None = None
    stager_key: int = 0


    def __init__(self,
                 layer: nn.Linear):

        self.layer = layer
        self.device = layer.weight.device

        self.rows = self.layer.weight.data.shape[1]
        self.columns = self.layer.weight.data.shape[0]

        # self.weights = self.layer.weight.data.T.clone().float().contiguous()
        self.weights = None
        self.hessian = None
        self.num_samples = 0
        self.num_batches = 0

        self.perm = None
        self.perm_cpu = None
        self.inv_perm = None

        self.scale = None
        self.qscale = None
        self.qscale_max = None
        self.qweight = None
        self.qgroups = None

        self.quant_device = 0
        self.hessian_device = 0
        self.output_device = 0
        self.target_device = None
        self.stager = None
        self.stager_key = 0

        # EXL2.5 options (set by conversion.quantize._configure_exl25 from job["exl25"]).
        # had: K-side Hadamard block (0 = off). Quantization runs in the kernel's stored
        # space: activations x @ H_int, weights H_int @ W / block. awq: per-channel
        # protection scale derived from the calibration Hessian diagonal. uniform_bits:
        # single bit-width fast lane (identity permutation, uniform groups).
        self.exl25_had = 0
        self.exl25_awq = False
        self.exl25_identity_perm = False
        self.exl25_uniform_bits = 0
        self.exl25_groupsize = 0
        self.awq_inv_scale = None


    def out_dev(self) -> torch.device:
        """Device that packed/scaled outputs are gathered onto. quant-plan.md §6.4 (M2)."""
        return torch.device(self.output_device)


    def apply_dev(self) -> torch.device:
        """
        Device the dequantized weight is written back onto.

        This is the owning module's device, not the worker device. The multi-GPU scheduler only spreads the
        quantization *work*; the model's device map is untouched, so a weight left on the worker device makes the
        post-quantization forward pass fail with "mat2 is on cuda:1, different from other tensors on cuda:0".
        """
        return torch.device(self.target_device if self.target_device is not None else self.output_device)


    def drop_buffers(self):

        self.perm = None
        self.perm_cpu = None
        self.invperm = None
        # self.g_idx = None
        self.scale = None
        self.qscale = None
        self.qscale_max = None
        self.qweight = None
        self.qgroups = None
        self.quant = None
        self.weights = None
        self.hessian = None
        self.hessian_inv = None

        gc.collect()
        torch.cuda.empty_cache()


    def configure(self,
                  group_size: dict,
                  bits = None,
                  bits_prop = None,
                  scale_bits: int = 4
                  ):

        self.group_size = group_size
        self.scale_bits = scale_bits
        self.bits = bits

        assert isinstance(bits, list)
        assert isinstance(bits_prop, list)
        assert sum(bits_prop) == 1

        groups = 0
        remaining_rows = self.rows
        self.bits_groups = []
        for b, p in zip(self.bits, bits_prop):
            assert p > 0
            gsz = self.group_size[b]
            g = math.ceil(min(self.rows * p, remaining_rows) / gsz)
            groups += g
            remaining_rows -= g * gsz
            self.bits_groups.append(g)

        assert remaining_rows <= 0

        self.total_groups = groups


    def add_batch(self, inputs):

        with torch.inference_mode():

            # dim = inputs.shape[-1]
            # inputs = inputs.view((-1, dim)).float().T.to("cuda:0")
            # ns = 1
            #
            # if self.hessian is None:
            #     self.hessian = torch.zeros((dim, dim), device = self.device, dtype = torch.float)
            # else:
            #     self.hessian.mul_(self.num_samples / (self.num_samples + ns))
            # self.num_samples += ns
            # inputs.mul_(math.sqrt(2 / self.num_samples))
            # self.hessian.addmm_(inputs, inputs.T)

            # self.num_batches += 1
            # num_samples = len(inputs)
            # # inputs = torch.cat(inputs, dim = 0)
            # inputs = inputs.view((-1, inputs.shape[-1])).float().T.to("cuda:0")
            # inputs *= math.sqrt(2 / num_samples)
            # self.hessian += inputs.matmul(inputs.T)

            if self.hessian is None:
                # Accumulate on the Hessian device, which the multi-GPU scheduler sets to the primary.
                # Defaults to 0, so single-GPU behaviour is unchanged. quant-plan.md §6.4.
                self.hessian = torch.zeros((self.rows, self.rows), device = torch.device(self.hessian_device),
                                           dtype = torch.float)

            self.num_batches += 1
            inputs = inputs.view((-1, inputs.shape[-1])).float().to(self.hessian.device)
            if self.exl25_had:
                from exllamav2.hadamard.hadamard import had_rotate_activations
                inputs = had_rotate_activations(inputs, self.exl25_had)
            self.hessian.addmm_(inputs.T, inputs)


    def stage_weights(self, stager: WeightStager | None = None, key: int = 0):
        """
        Load the layer's weights into ``self.weights`` as a contiguous fp32 host tensor of shape
        ``(in_features, out_features)``, then apply the activation-order permutation.

        Split out of ``prepare()`` so that the multi-GPU scheduler can run the Hessian half once per distinct
        ``H`` on the primary device and the weight half independently per tensor. quant-plan.md §6.4 (M2).

        When ``stager`` is supplied the device-to-host copy is issued through pinned memory on a side stream
        (quant-plan.md §6.8, M5). The result is bit-identical either way; only the wall-clock cost differs.
        """

        with torch.inference_mode():

            if stager is not None:
                self.weights = stager.get(key, self.layer.weight.data)
            else:
                self.weights = self.layer.weight.data.cpu().T.clone().float().contiguous()

            # EXL2.5: move the weights into the stored space, W' = H_int @ (S @ W) / block.
            # The AWQ scale comes from prepare_hessian (must run first); the Hadamard
            # rotation is an involution up to 1/block, so the same call un-rotates in
            # apply_quant.
            if self.awq_inv_scale is not None:
                self.weights *= (1.0 / self.awq_inv_scale).to(self.weights.device)[:, None]
            if self.exl25_had:
                from exllamav2.hadamard.hadamard import had_rotate_weights
                self.weights = had_rotate_weights(self.weights, self.exl25_had).contiguous()

            # Zero weights that have no impact. Disabling this since it feels a little drastic based on just the calibration
            # data. It likely never triggers, anyway.

            # dead = diagonal == 0.0
            # self.hessian[dead, dead] = 1
            # self.weights[dead, :] = 0

            # Activation order

            if self.perm_cpu is not None:
                self.weights = self.weights[self.perm_cpu, :]


    def prepare_hessian(self, no_h_inv = False):
        """
        Normalize ``H``, pick the activation order, damp, and invert. Sets ``self.perm``, ``self.perm_cpu``
        and ``self.hessian_inv``, and drops ``self.hessian``.

        **This mutates the Hessian in place** (``hessian[d, d] += damp``) and then releases the reference, so it
        must be called exactly once per distinct ``H``. q/k/v share one ``hessian_inv`` via ``reuse_h``, so calling
        this concurrently on three threads would race. quant-plan.md §6.4 / R11.
        """

        with torch.inference_mode():

            if self.hessian is None or self.num_batches == 0:
                self.hessian = torch.eye(self.rows, device = self.device, dtype = torch.float)
            else:
                self.hessian /= self.num_batches

            # EXL2.5: move H into the stored (rotated) activation space, then derive the
            # AWQ protection scale from its diagonal and congruence-scale H by S^-1.
            if self.exl25_had:
                from exllamav2.hadamard.hadamard import had_rotate_hessian
                self.hessian = had_rotate_hessian(self.hessian, self.exl25_had)
            diagonal = torch.diag(self.hessian)
            if self.exl25_awq:
                rms = diagonal.clamp_min(1e-12).sqrt()
                s = (rms / rms.mean()).pow(0.5).clamp(0.25, 4.0)
                self.awq_inv_scale = 1.0 / s
                self.hessian = self.hessian / s[:, None] / s[None, :]
                diagonal = torch.diag(self.hessian)

            # Activation order (identity for the EXL2.5 fast lane: Hadamard makes every
            # group tight, so act-order permutation is redundant and the kernel drops it)

            if self.exl25_identity_perm:
                self.perm = torch.arange(self.rows, device = self.hessian.device)
            else:
                self.perm = torch.argsort(diagonal, descending = True)
            self.perm_cpu = self.perm.cpu()

            if self.hessian.numel() > 6e8:
                hessian_cpu = self.hessian.cpu()
                self.hessian = None
                hessian = hessian_cpu[self.perm_cpu][:, self.perm_cpu]
                hessian = hessian.to(torch.device(self.hessian_device))
            else:
                # perm lives on the layer's device; the Hessian may not. Index on the Hessian's device.
                perm = self.perm.to(self.hessian.device)
                hessian = self.hessian[perm][:, perm]
                self.hessian = None

            # In case numerical errors have caused some asymmetry in H, assume it's close to symmetrical and force it.
            # (Doesn't seem to be needed)

            # torch.cuda.empty_cache()
            # hessian = (hessian + hessian.T) * 0.5
            # torch.cuda.empty_cache()

            # Damping

            diagonal = torch.diag(hessian)
            damp = torch.clamp(self.percdamp * torch.mean(diagonal), min = 1e-5)

            # Inverse of H

            attempts = 0
            while not no_h_inv:

                try:

                    # Index on the Hessian's device, not the layer's. These differ once the multi-GPU
                    # scheduler assigns a Hessian device other than the layer's own device; the old code
                    # only worked because both were cuda:0. quant-plan.md §6.4.
                    d = torch.arange(self.rows, device = hessian.device)
                    hessian[d, d] += damp

                    current_device = self.hessian_device
                    max_devices = torch.cuda.device_count()

                    done = False
                    fail_device = False
                    while not done:
                        try:
                            hessian = hessian.to(torch.device(current_device))

                            hessian_inv = torch.linalg.cholesky(hessian)
                            hessian_inv = torch.cholesky_inverse(hessian_inv)

                            # The Cholesky inverse will sometimes fail to compute due to accumulated rounding errors when H
                            # is very large (e.g. 70B MLP down proj) and a lot of calibration data is used (e.g. 100 rows of
                            # 4096 tokens). This won't always throw an exception and sometimes just results in a NaN tensor.

                            if torch.any(torch.isnan(hessian_inv)): raise RuntimeError

                            # Test inversion

                            hessian_inv = torch.linalg.cholesky(hessian_inv, upper = True)
                            hessian_inv = hessian_inv.contiguous()

                            done = True
                            break

                        except torch.cuda.OutOfMemoryError as e:
                            current_device += 1
                            print(f" !! Out of memory (H), moving to device {current_device}")
                            if current_device == max_devices:
                                raise e
                            self.hessian_device = current_device

                    if done: break

                except RuntimeError as runtime_error:

                    if "out of memory" in str(runtime_error):
                        raise runtime_error

                    # If inverting failed, assume there were non-positive eigenvalues, so apply more damping to shift
                    # the eigenvalues in a positive direction.

                    print(" !! Warning: Applied additional damping")

                    attempts += 1
                    if attempts == 10:
                        raise ValueError("Hessian is not invertible")

            # Swap H to system RAM

            self.hessian_inv = None if no_h_inv else hessian_inv.cpu()
            self.hessian = None


    def prepare(self, no_h_inv = False, stager: WeightStager | None = None, key: int = 0):
        """
        Full preparation: Hessian inversion followed by weight staging. This is the historical entry point and
        behaves exactly as before when called with no arguments.
        """

        with torch.inference_mode():
            self.prepare_hessian(no_h_inv)
            self.stage_weights(stager, key)


    def reuse_h_hessian(self, other):
        """
        Adopt ``other``'s inverted Hessian and activation order, without touching weights.

        Split out of ``reuse_h()`` so the multi-GPU scheduler can do all Hessian work serially on the primary
        device and all weight work in parallel worker threads. quant-plan.md §6.4 (M2).
        """

        with torch.inference_mode():
            self.hessian_inv = other.hessian_inv
            self.hessian = None
            self.perm = other.perm
            self.perm_cpu = other.perm_cpu


    def reuse_h(self, other, stager: WeightStager | None = None, key: int = 0):

        with torch.inference_mode():

            self.reuse_h_hessian(other)

            # Prepare weights

            self.stage_weights(stager, key)


    def quantize_rtn_inplace(self, keep_qweight = False, apply = False):
        assert apply and keep_qweight
        assert not (self.exl25_had or self.exl25_awq or self.exl25_uniform_bits), \
            "EXL2.5 extensions are not supported on the RTN path (large lm_head); keep it legacy"

        with torch.inference_mode():

            current_device = self.quant_device
            max_devices = torch.cuda.device_count()

            weights_cpu = self.weights.cpu().contiguous()

            done = False
            while not done:

                try:

                    weights = weights_cpu.to(torch.device(current_device))
                    self.qweight = torch.zeros_like(weights, dtype = torch.short, device = torch.device(current_device))

                    num_groups = 0
                    for bits_idx in range(len(self.bits)):
                        num_groups += self.bits_groups[bits_idx]

                    scale = []
                    qscale = []
                    qscale_max = torch.empty((num_groups,), dtype = torch.float, device = torch.device(current_device))
                    qgroups = []

                    group_idx = 0
                    group_idx_list = []

                    b = 0
                    for bits_idx, bits in enumerate(self.bits):
                        quantizer = AdaptiveQuantizer(bits = bits, scale_bits = self.scale_bits)

                        for group in range(self.bits_groups[bits_idx]):
                            a = b
                            b = min(a + self.group_size[bits], self.rows)

                            qgroups.append(bits)
                            qgroups.append(0)

                            quantizer.find_params(weights[a : b, :])
                            scale.append(quantizer.scale)
                            qscale.append(quantizer.qscale)
                            qscale_max[group_idx] = quantizer.qscale_max

                            ext_c.quantize_range_inplace(weights,
                                                         quantizer.scale,
                                                         self.qweight,
                                                         quantizer.qzero,
                                                         quantizer.maxq,
                                                         a,
                                                         b)

                            group_idx_list += [group_idx] * (b - a)
                            group_idx += 1

                    done = True

                except torch.cuda.OutOfMemoryError as e:
                    current_device += 1
                    print(f" !! Out of memory (Q), moving to device {current_device}")
                    if current_device == max_devices:
                        raise e
                    self.quant_device = current_device

            # Create g_idx to store inverse activation order

            self.invperm = torch.argsort(self.perm).to(self.out_dev())

            # Store scales

            self.scale = torch.stack(scale, dim = 0).to(self.out_dev())
            self.qscale = torch.stack(qscale, dim = 0).to(self.out_dev())
            self.qscale_max = qscale_max.to(torch.float16).to(self.out_dev())
            self.qgroups = torch.tensor(qgroups, dtype = torch.short).to(self.out_dev())

            # I love Python

            scale = None
            qscale = None
            qscale_max = None
            qgroups = None
            group_idx_list = None

            qc = weights.cpu()
            qc = qc.to(torch.half)
            invperm = self.invperm.cpu()
            q = qc[invperm, :].T
            q = q.reshape(weights.T.shape)

            dev = weights.device
            weights = None
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            q = q.to(dev)
            self.layer.weight.data = q
            self.weights = q.T


    def quantize(self, keep_qweight = False, apply = False):

        with torch.inference_mode():

            current_device = self.quant_device
            max_devices = torch.cuda.device_count()

            weights_cpu = self.weights.cpu()

            done = False
            while not done:

                try:

                    hessian_inv_cuda = self.hessian_inv.to(torch.device(current_device))

                    # if apply:
                    #     weights = self.weights
                    #     self.layer.weight.data = torch.zeros((1, 1), dtype = torch.float32, device = weights.device)
                    # else:
                    #     weights = self.weights.clone()
                    weights = weights_cpu.to(torch.device(current_device))

                    self.quant = torch.zeros_like(weights_cpu, device = torch.device(current_device))

                    if keep_qweight:
                        self.qweight = torch.zeros_like(weights, dtype = torch.short)

                    # Quantize groups

                    num_groups = 0
                    for bits_idx in range(len(self.bits)):
                        num_groups += self.bits_groups[bits_idx]

                    scale = []
                    qscale = []
                    qscale_max = torch.empty((num_groups,), dtype = torch.float, device = torch.device(current_device))
                    qgroups = []

                    error = weights.clone()
                    group_idx = 0
                    group_idx_list = []

                    b = 0
                    for bits_idx, bits in enumerate(self.bits):
                        quantizer = AdaptiveQuantizer(bits = bits, scale_bits = self.scale_bits)

                        for group in range(self.bits_groups[bits_idx]):
                            a = b
                            b = min(a + self.group_size[bits], self.rows)

                            qgroups.append(bits)
                            qgroups.append(0)

                            quantizer.find_params(weights[a : b, :])
                            scale.append(quantizer.scale)
                            qscale.append(quantizer.qscale)
                            qscale_max[group_idx] = quantizer.qscale_max

                            ext_c.quantize_range(self.quant,
                                                 quantizer.scale,
                                                 self.qweight if keep_qweight else none_tensor,
                                                 quantizer.qzero,
                                                 quantizer.maxq,
                                                 hessian_inv_cuda,
                                                 weights,
                                                 error,
                                                 a,
                                                 b)

                            group_idx_list += [group_idx] * (b - a)
                            group_idx += 1

                    done = True

                except torch.cuda.OutOfMemoryError as e:
                    current_device += 1
                    print(f" !! Out of memory (Q), moving to device {current_device}")
                    if current_device == max_devices:
                        raise e
                    self.quant_device = current_device

            # Create g_idx to store inverse activation order

            # self.g_idx = torch.tensor(group_idx_list, dtype = torch.int32, device = self.device)
            # self.g_idx = torch.tensor(group_idx_list, dtype = torch.int32)

            self.quant = self.quant.to(self.out_dev())
            self.invperm = torch.argsort(self.perm).to(self.out_dev())
            # self.g_idx = self.g_idx[self.invperm]

            # Store scales

            self.scale = torch.stack(scale, dim = 0).to(self.out_dev())
            self.qscale = torch.stack(qscale, dim = 0).to(self.out_dev())
            self.qscale_max = qscale_max.to(torch.float16).to(self.out_dev())
            self.qgroups = torch.tensor(qgroups, dtype = torch.short).to(self.out_dev())

            # I love Python

            weights = None
            error = None
            scale = None
            qscale = None
            qscale_max = None
            qgroups = None
            group_idx_list = None

            # Apply

            if apply:
                self.apply_quant()


    def quant_error(self):

        with torch.inference_mode():

            # EXL2.5 stores rotated/scaled weights; the raw-space comparison below is
            # meaningless there (the conversion sanity check uses the transformed-identity
            # forward instead).
            if self.exl25_had or self.awq_inv_scale is not None:
                return 0.0, 0.0, 0.0

            q = self.quant[self.invperm, :]
            diff = torch.abs(q - self.layer.weight.data.T)
            mat_error_1 = (diff > 0.01).sum().item() / diff.numel()
            mat_error_5 = (diff > 0.05).sum().item() / diff.numel()
            mat_error_10 = (diff > 0.10).sum().item() / diff.numel()
            return mat_error_1, mat_error_5, mat_error_10


    def stored_to_logical(self, w):
        """
        EXL2.5 inverse transform, w in (k, n) packed/dequant space -> logical weights.
        stage_weights stores W' = H_int @ (S @ W) / block, so the inverse is
        W = S^-1 @ (H_int @ W'): un-rotate FIRST with NO /block (H_int^2 = block*I
        cancels it), then undo the AWQ scale. Order matters: H_int and the diagonal
        S do not commute. No-op for legacy tensors.
        """
        if self.exl25_had:
            from exllamav2.hadamard.hadamard import _had_int
            b = self.exl25_had
            k, n = w.shape
            h = _had_int(b, w.device, w.dtype)
            w = torch.matmul(h, w.reshape(k // b, b, n)).reshape(k, n)
        if self.awq_inv_scale is not None:
            w = w * self.awq_inv_scale.to(device = w.device, dtype = w.dtype)[:, None]
        return w


    def apply_quant(self):

        self.hessian = None

        qc = self.quant.cpu()
        invperm = self.invperm.cpu()
        q = qc[invperm, :].T

        # EXL2.5: back to logical space so the live model (and every later calibration
        # forward) keeps using ordinary weights. q is (n, k).
        if self.exl25_had or self.awq_inv_scale is not None:
            q = self.stored_to_logical(q.T).T

        q = q.reshape(self.quant.T.shape)

        gc.collect()
        torch.cuda.empty_cache()

        q = q.to(self.apply_dev())
        self.layer.weight.data = q


    def apply_temp(self):

        q = self.quant[self.invperm, :].T
        if self.exl25_had or self.awq_inv_scale is not None:
            q = self.stored_to_logical(q.T).T
        temp_layer = nn.Linear(self.layer.in_features, self.layer.out_features, False, device = "meta", dtype = torch.float16)
        temp_layer.weight = nn.Parameter(q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data))
        return temp_layer


    def pack(self, key, qparams):

        self.qgroups = self.qgroups.to(self.out_dev())
        # self.qscale_max = self.qscale_max.to("cude:0")

        assert qparams.scale_bits in [4]
        # assert self.columns % 32 == 0

        output = {}
        output[key + ".q_invperm"] = self.invperm.to(torch.int)
        output[key + ".q_scale_max"] = self.qscale_max
        output[key + ".q_groups"] = self.qgroups
        if self.layer.bias is not None:
            output[key + ".bias"] = self.layer.bias.data

        columns = self.columns
        rem_rows = self.rows
        padding = -columns % 32

        if padding != 0:
            print(f" !! Note: Padding quantized tensor {key}")
            qst = F.pad(self.qscale, (0, padding)).contiguous()
            qwt = F.pad(self.qweight, (0, padding)).contiguous()
        else:
            qst = self.qscale
            qwt = self.qweight

        qst_packed = torch.zeros((qst.shape[0], qst.shape[1] * qparams.scale_bits // 32), dtype = torch.int32, device = self.out_dev())
        if qparams.scale_bits == 4: ext_c.pack_rows_4(qst, qst_packed)
        output[key + ".q_scale"] = qst_packed

        qwt_packed = []

        i = 0
        row = 0
        out_row = 0
        while i < self.qscale.shape[0]:

            bits = self.qgroups[i * 2].item()
            self.qgroups[i * 2 + 1] = out_row
            i += 1

            rows = min(self.group_size[bits], rem_rows)
            wpqr = 32 / bits
            qrows = rows / wpqr
            assert i == self.qgroups.shape[-1] or qrows == int(qrows)
            qrows = math.ceil(qrows)

            g_qwt = qwt[row:row+rows, :].contiguous()
            g_qwt_packed = torch.zeros((qrows, columns + padding), dtype = torch.int32, device = g_qwt.device)

            if padding > 0: g_qwt[:, -padding:] = 2 ** (bits - 1)

            ext_c.pack_columns(g_qwt, g_qwt_packed, bits)
            qwt_packed.append(g_qwt_packed)

            # print(row, rows, bits)

            row += rows
            out_row += qrows
            rem_rows -= rows


        qwt_packed = torch.cat(qwt_packed, dim = 0)
        output[key + ".q_weight"] = qwt_packed

        # EXL2.5 tags (self-describing; absent = legacy EXL2). q_groups/q_group_map and
        # the identity q_invperm stay in the file, so the legacy kernel remains a valid
        # fallback for any shape the fast lane can't serve.
        if self.exl25_uniform_bits:
            output[key + ".q_uniform_bits"] = torch.tensor(self.exl25_uniform_bits, dtype = torch.int32)
            output[key + ".q_group_size"] = torch.tensor(self.exl25_groupsize, dtype = torch.int32)
            if self.exl25_had:
                output[key + ".q_hadamard"] = torch.tensor(self.exl25_had, dtype = torch.int32)
            if self.awq_inv_scale is not None:
                output[key + ".q_awq_scale"] = self.awq_inv_scale.to(torch.float16).to(self.out_dev())

        return output



