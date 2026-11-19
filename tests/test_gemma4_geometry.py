"""
Gemma-4 config / geometry / rope unit tests.

PLAN.md Stage B steps 7-9 and the §7 verification checklist. These tests do NOT need a
GPU for the config and geometry parts; the rope table tests do, because
``prepare_sincos`` builds tensors on a CUDA device.

Run with:
    pytest tests/test_gemma4_geometry.py -v
"""

import json
import os

import pytest
import torch


# ---------------------------------------------------------------------------------------------
# Fixtures: a synthetic Gemma-4-31B text_config
# ---------------------------------------------------------------------------------------------

GEMMA4_31B_TEXT_CONFIG = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "text_config": {
        "num_hidden_layers": 60,
        "hidden_size": 5376,
        "intermediate_size": 21504,
        "head_dim": 256,
        "global_head_dim": 512,
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "num_global_key_value_heads": 4,
        "attention_k_eq_v": True,
        "sliding_window": 1024,
        "final_logit_softcapping": 30.0,
        "hidden_activation": "gelu_pytorch_tanh",
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": True,
        "vocab_size": 262144,
        "max_position_embeddings": 262144,
        "use_bidirectional_attention": "vision",
        "enable_moe_block": False,
        "hidden_size_per_layer_input": 0,
        "rope_parameters": {
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        },
        # 5 sliding then 1 full, repeating: layers 5, 11, 17, ... 59 are full.
        "layer_types": [
            "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
            for i in range(60)
        ],
    },
}


@pytest.fixture
def gemma4_dir(tmp_path):
    """Write the synthetic config to a temp dir so ExLlamaV2Config can read it."""
    d = tmp_path / "gemma4-31b"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(GEMMA4_31B_TEXT_CONFIG))
    return str(d)


def _cfg(gemma4_dir):
    from exllamav2.config import ExLlamaV2Config
    cfg = ExLlamaV2Config()
    cfg.model_dir = gemma4_dir
    cfg.prepare(no_tensors = True)
    return cfg


def _device_context(model):
    """
    A prepared device context for device 0.

    `ExLlamaV2.set_device_map()` only creates a context for devices that actually receive
    modules, so a model that was constructed but never loaded has `device_context == []` and
    `get_device_context(0)` raises IndexError. These tests only need the sin/cos tables, which
    `prepare_sincos()` builds from the config alone, so construct the context directly.
    """
    from exllamav2.device import ExLlamaV2DeviceContext

    if not model.device_context:
        model.device_context.append(ExLlamaV2DeviceContext(model, 0, 1))
    ctx = model.get_device_context(0, scratch = False)
    ctx.prepare_sincos()
    return ctx


# ---------------------------------------------------------------------------------------------
# Config reads
# ---------------------------------------------------------------------------------------------

def test_config_reads(gemma4_dir):
    cfg = _cfg(gemma4_dir)
    assert cfg.head_dim == 256
    assert cfg.global_head_dim == 512
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 16
    assert cfg.num_global_key_value_heads == 4
    assert cfg.attention_k_eq_v is True
    assert cfg.sliding_window == 1024
    assert cfg.rope_type_global == "proportional"
    assert cfg.rope_partial_rotary_factor_global == 0.25
    assert cfg.rope_theta_global == 1e6
    assert cfg.rope_theta_local == 10000.0
    assert len(cfg.layer_types) == 60


def test_layer_types_pattern(gemma4_dir):
    """The 5:1 pattern must reproduce layer_types exactly. PLAN.md §2.2 step 3."""
    cfg = _cfg(gemma4_dir)
    full = [i for i in range(60) if cfg.layer_types[i] == "full_attention"]
    assert full == [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
    assert len(full) == 10


def test_per_layer_geometry(gemma4_dir):
    """PLAN.md §2.2 step 1 / §7."""
    cfg = _cfg(gemma4_dir)
    for i in range(60):
        head_dim, nq, nkv, is_full = cfg.get_layer_attn_geometry(i)
        assert nq == 32
        if is_full:
            assert head_dim == 512
            assert nkv == 4
        else:
            assert head_dim == 256
            assert nkv == 16
        assert nq % nkv == 0


def test_uniform_geometry_unchanged():
    """Non-Gemma-4 models must keep the uniform geometry. PLAN.md §2.2 step 1."""
    from exllamav2.config import ExLlamaV2Config

    cfg = ExLlamaV2Config()
    cfg.layer_types = None
    cfg.head_dim = 128
    cfg.num_attention_heads = 32
    cfg.num_key_value_heads = 8

    for i in range(4):
        assert cfg.get_layer_attn_geometry(i) == (128, 32, 8, True)


def test_sincos_size_is_full_head_dim(gemma4_dir):
    """
    The single easiest bug in the whole port: sincos_size for proportional rope must be
    the full head_dim (512), not head_dim * partial_rotary_factor (128).
    PLAN.md §2.3 / §8 trap 1.
    """
    cfg = _cfg(gemma4_dir)

    # Full-attention layer: proportional rope -> full 512-wide table.
    assert cfg.get_layer_sincos_size(5) == 512
    # Sliding layer: default rope -> 256-wide table.
    assert cfg.get_layer_sincos_size(0) == 256


def test_moe_and_per_layer_input_are_rejected(tmp_path):
    """Unsupported features must fail loudly. PLAN.md §2.1 step 4."""
    from exllamav2.config import ExLlamaV2Config

    for key, value in [("enable_moe_block", True), ("hidden_size_per_layer_input", 128)]:
        bad = json.loads(json.dumps(GEMMA4_31B_TEXT_CONFIG))
        bad["text_config"][key] = value
        d = tmp_path / key
        d.mkdir()
        (d / "config.json").write_text(json.dumps(bad))

        cfg = ExLlamaV2Config()
        cfg.model_dir = str(d)
        with pytest.raises(AssertionError):
            cfg.prepare(no_tensors = True)


# ---------------------------------------------------------------------------------------------
# Rope tables (needs CUDA: prepare_sincos builds tensors on the device)
# ---------------------------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason = "prepare_sincos needs CUDA")
def test_proportional_rope_table(gemma4_dir):
    """
    Global-layer sin/cos must be [1, 1, max_seq_len, 512], and the non-rotated region
    (the last 384 columns) must be exactly sin = 0, cos = 1.
    PLAN.md §2.3 verification.
    """
    from exllamav2.model import ExLlamaV2

    cfg = _cfg(gemma4_dir)
    cfg.max_seq_len = 64
    model = ExLlamaV2(cfg)
    ctx = _device_context(model)

    sin = ctx.sin[0]
    cos = ctx.cos[0]

    assert sin.shape == (1, 1, 64, 512), f"got {tuple(sin.shape)}"
    assert cos.shape == (1, 1, 64, 512), f"got {tuple(cos.shape)}"

    # rope_angles = int(0.25 * 512 // 2) = 64, so 64 of the 256 frequencies are rotated and
    # 192 are zero-filled. The zero tail is NOT contiguous in the final 512-wide table, because
    # NEOX builds `emb = cat((freqs, freqs))`: the rotated frequencies appear twice, at
    # [0:64] and again at [256:320], and the zero frequencies appear at [64:256] and [320:512].
    #
    # Verified numerically, not assumed: nonzero sin columns are exactly 0..63 and 256..319.
    # An earlier version of this test asserted `sin[..., 128:] == 0`, which is the GPTJ layout
    # (`repeat_interleave`) and is wrong for Gemma-4, which uses NEOX.
    # `any`, not `all`: row 0 is position 0, where freqs == 0 and therefore sin == 0 for every
    # column. Only the *column* pattern is being asserted here.
    assert torch.any(sin[..., 0:64] != 0), "first rotated block must be nonzero"
    assert torch.all(sin[..., 64:256] == 0), "zero-freq block [64:256] must be sin = 0"
    assert torch.any(sin[..., 256:320] != 0), "second rotated block must be nonzero"
    assert torch.all(sin[..., 320:] == 0), "zero-freq block [320:] must be sin = 0"
    assert torch.all(cos[..., 64:256] == 1), "zero-freq block [64:256] must be cos = 1"
    assert torch.all(cos[..., 320:] == 1), "zero-freq block [320:] must be cos = 1"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "prepare_sincos needs CUDA")
def test_sliding_rope_table(gemma4_dir):
    """Sliding-layer sin/cos must be [1, 1, max_seq_len, 256]. PLAN.md §2.3."""
    from exllamav2.model import ExLlamaV2

    cfg = _cfg(gemma4_dir)
    cfg.max_seq_len = 64
    model = ExLlamaV2(cfg)
    ctx = _device_context(model)

    assert ctx.sin[1].shape == (1, 1, 64, 256)
    assert ctx.cos[1].shape == (1, 1, 64, 256)
