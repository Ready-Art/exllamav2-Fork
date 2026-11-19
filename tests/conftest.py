"""
pytest configuration for the exllamav2 test suite.

The `test_m3_*.py` files are **probe scripts, not pytest modules**. They execute at import
time: they load the full 62 GB Gemma-4 checkpoint, build a second module replica on another
device, and print a comparison. That is deliberate — they were written to answer one question
each during the multi-GPU work (quant-plan.md §6.5) and are meant to be run directly:

    python tests/test_m3_shard_hessian.py

Collecting them under pytest is wrong for two reasons:
  1. Import-time execution means `pytest --collect-only` loads the checkpoint.
  2. They need a checkpoint that most machines do not have, so collection fails with
     `IndexError` / `FileNotFoundError` rather than a skip.

They are therefore excluded from collection unless EXL2_RUN_M3_PROBES=1 is set.
"""

import os


def pytest_addoption(parser):
    """
    Options shared by the tests that need a converted checkpoint.

    These live here, not in the test modules. pytest only honours `pytest_addoption` in a plugin or
    conftest.py; a definition inside a test module is silently ignored, so `--model-dir` was never
    registered and every test that read it errored with "no option named '--model-dir'".
    """
    parser.addoption(
        "--model-dir", action = "store", default = None,
        help = "Path to a converted Gemma-4 exl2 model (enables the SWA parity tests)",
    )
    parser.addoption(
        "--hf-model", action = "store", default = None,
        help = "Path to a tiny HF checkpoint to convert (enables the hb16 parity test)",
    )
    parser.addoption(
        "--work-dir", action = "store", default = "/tmp/hb16-test",
        help = "Scratch directory for converted models",
    )


collect_ignore_glob = []

if os.environ.get("EXL2_RUN_M3_PROBES") != "1":
    collect_ignore_glob += [
        "test_m3_device_context.py",
        "test_m3_replica_probe.py",
        "test_m3_shard_correctness.py",
        "test_m3_shard_hessian.py",
    ]
