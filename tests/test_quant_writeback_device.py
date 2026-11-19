"""
Regression test for Stage K bug 8: the multi-GPU quantizer wrote the dequantized weight back onto the worker
device instead of the owning module's device.

The bug only appears in the *quantize* pass, never in the measurement pass, because the measurement pass runs each
module on its own device. That is why it survived the whole measurement phase and aborted at step 183/183.

Run with:  python tests/test_quant_writeback_device.py
Needs 2 GPUs for the multi-device case; the single-device case runs anywhere with CUDA.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from exllamav2.conversion.adaptivegptq import AdaptiveGPTQ


def check(name, cond, detail = ""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        raise AssertionError(name)


def test_default_falls_back_to_output_device():
    """With no target_device set, apply_dev() must equal out_dev(), i.e. single-GPU behaviour is unchanged."""
    layer = nn.Linear(8, 8, bias = False, device = "cuda:0", dtype = torch.half)
    q = AdaptiveGPTQ(layer)
    q.output_device = 0
    check("target_device defaults to None", q.target_device is None)
    check("apply_dev() == out_dev() when unset",
          q.apply_dev() == q.out_dev(), f"{q.apply_dev()} vs {q.out_dev()}")


def test_apply_dev_follows_target_device():
    """apply_dev() must follow target_device, not output_device, once the scheduler sets it."""
    layer = nn.Linear(8, 8, bias = False, device = "cuda:0", dtype = torch.half)
    q = AdaptiveGPTQ(layer)
    q.output_device = 0
    q.target_device = 0
    check("apply_dev() honours target_device", q.apply_dev() == torch.device("cuda:0"),
          str(q.apply_dev()))

    if torch.cuda.device_count() < 2:
        print("SKIP  worker != owner case (needs 2 GPUs)")
        return

    q.output_device = 1          # packed tensors gathered on the worker
    q.target_device = 0          # weight belongs to the module on device 0
    check("apply_dev() != out_dev() when they differ",
          q.apply_dev() != q.out_dev(), f"{q.apply_dev()} vs {q.out_dev()}")
    check("apply_dev() is the owner device", q.apply_dev() == torch.device("cuda:0"),
          str(q.apply_dev()))


def test_apply_quant_lands_on_target_device():
    """
    The real check: run apply_quant() with the weight staged on the worker device and assert the resulting
    `layer.weight.data` is on the owner device. This is the exact failure that aborted the Stage K run.
    """
    if torch.cuda.device_count() < 2:
        print("SKIP  apply_quant device placement (needs 2 GPUs)")
        return

    owner, worker = 0, 1
    layer = nn.Linear(16, 16, bias = False, device = f"cuda:{owner}", dtype = torch.half)

    q = AdaptiveGPTQ(layer)
    q.output_device = worker
    q.target_device = owner

    # Minimal state apply_quant() needs: `quant` is the dequantized matrix, shape (in, out) after the transpose
    # dance, and `invperm` is the identity.
    q.quant = torch.randn(16, 16, device = f"cuda:{worker}", dtype = torch.float)
    q.invperm = torch.arange(16, device = f"cuda:{worker}")

    q.apply_quant()

    check("weight is on the owner device",
          layer.weight.data.device == torch.device(f"cuda:{owner}"),
          str(layer.weight.data.device))
    check("weight is not on the worker device",
          layer.weight.data.device != torch.device(f"cuda:{worker}"),
          str(layer.weight.data.device))

    # And the post-quantization forward pass must now work: activation on the owner device, weight on the owner device.
    #
    # `apply_quant()` leaves the weight in fp32; that is transient and pre-existing, not part of this bug. In the real
    # pipeline `quant_linear()` overwrites `source.linear.weight.data` with the half-precision reconstruction a few
    # lines later. Mirror that here so the forward pass exercises the device question and not a dtype one.
    layer.weight.data = layer.weight.data.to(torch.half)
    x = torch.randn(4, 16, device = f"cuda:{owner}", dtype = torch.half)
    try:
        y = layer(x)
        ok = True
        detail = str(tuple(y.shape))
    except RuntimeError as e:
        ok = False
        detail = str(e)
    check("forward pass after apply_quant succeeds", ok, detail)


if __name__ == "__main__":
    test_default_falls_back_to_output_device()
    test_apply_dev_follows_target_device()
    test_apply_quant_lands_on_target_device()
    print("\nall write-back device tests passed")
