"""
Concurrency smoke test for the fork MP+MTP multi-batch slot scheduler (tabbyAPI server).

Drives a RUNNING tabbyAPI fork server (max_batch_size >= 4, MTP drafter) through the
phases the scheduler must survive:

  A  4 concurrent requests, two of which share a long prefix (exercises the
     cross-slot KV clone while the source slot is still decoding).
  B  a chat continuation of one finished request (per-slot prompt-cache hit/restore).
  C  4 concurrent requests with WILDLY different context lengths (exercises the
     per-(B, q_len) graph envelope recaptures + stable P2P epochs; the old global
     clear thrashed here and leaked arena slots).

Usage (server already up, e.g. port 5001 with disable_auth):
    venv/bin/python exllamav2-Fork/mtp/batch_smoke_client.py [--url http://127.0.0.1:5001]

Prints per-request tok/s + completion sanity, and the phase markers to grep for in
the SERVER log: "prompt cache clone", "prompt cache hit/restore", "[mtp-mp-batch]",
"[TP graph] captured", "[mtp-drafter] captured ... batch draft window graph".
Any request error / empty completion / server-side "[TP graphs] disabled" or
"[fi-guard] CORRUPTION" in the log = FAIL.
"""

import argparse
import asyncio
import json
import time

import aiohttp


FILLER = (
    "The history of computing is a long narrative of abstraction, from mechanical "
    "looms to punched cards, vacuum tubes, transistors, integrated circuits, and "
    "the vast distributed systems that now carry most of the world's information. "
    "Each layer hid the complexity of the one below it, and each new layer created "
    "new kinds of complexity of its own. Engineers learned to reason about systems "
    "they could not fully inspect, relying on invariants, tests, and careful "
    "measurement rather than complete understanding. "
)


def filler_tokens(n_chars: int) -> str:
    reps = n_chars // len(FILLER) + 1
    return (FILLER * reps)[:n_chars]


async def chat(session, url, messages, max_tokens, temperature, tag, top_p=0.9,
               stats=None):
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    t0 = time.perf_counter()
    text = ""
    n_chunks = 0
    first_tok = None
    async with session.post(url + "/v1/chat/completions", json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"[{tag}] HTTP {resp.status}: {body[:500]}")
        async for raw in resp.content:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "choices" not in obj:
                # tabby sends an error object mid-stream when generation aborts
                raise RuntimeError(f"[{tag}] server stream error: {json.dumps(obj)[:800]}")
            delta = obj["choices"][0].get("delta", {})
            piece = delta.get("content") or ""
            if piece:
                if first_tok is None:
                    first_tok = time.perf_counter() - t0
                text += piece
                n_chunks += 1
    dt = time.perf_counter() - t0
    # tabby streams one chunk per emitted token -- n_chunks is the token count
    print(f"[{tag}] {dt:.1f}s total, first-chunk {first_tok if first_tok else -1:.1f}s, "
          f"{n_chunks} tok, {n_chunks / max(dt, 1e-9):.1f} tok/s", flush=True)
    if not text.strip():
        raise RuntimeError(f"[{tag}] EMPTY completion")
    print(f"[{tag}] head: {text[:160]!r}", flush=True)
    if stats is not None:
        stats[tag] = (n_chunks, dt)
    return text


async def phase_a(url):
    print("=== PHASE A: 4 concurrent (2 share a long prefix -> clone) ===", flush=True)
    shared = "Reference document:\n" + filler_tokens(9000) + "\n\n"
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=900)) as s:
        results = await asyncio.gather(
            chat(s, url, [{"role": "user", "content":
                 "Write a short paragraph about the sea."}], 100, 0.7, "A1-sea"),
            chat(s, url, [{"role": "user", "content":
                 "List three prime numbers and explain briefly what a prime is."}],
                 100, 0.0, "A2-primes-greedy"),
            chat(s, url, [{"role": "user", "content":
                 shared + "Question: what did engineers rely on instead of complete "
                 "understanding? Answer in one sentence."}], 80, 0.7, "A3-doc-q1"),
            chat(s, url, [{"role": "user", "content":
                 shared + "Question: name two layers of abstraction mentioned in the "
                 "document. Answer in one sentence."}], 80, 0.7, "A4-doc-q2"),
        )
    return results


async def phase_b(url, prior_text):
    print("=== PHASE B: continuation (per-slot prompt cache hit/restore) ===", flush=True)
    msgs = [
        {"role": "user", "content": "Write a short paragraph about the sea."},
        {"role": "assistant", "content": prior_text},
        {"role": "user", "content": "Now rewrite that paragraph in exactly two sentences."},
    ]
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)) as s:
        return await chat(s, url, msgs, 100, 0.7, "B-continuation")


async def phase_c(url, long_chars=60000):
    print("=== PHASE C: 4 concurrent, mixed lengths (envelope recapture churn) ===",
          flush=True)
    long_ctx = filler_tokens(long_chars)
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1200)) as s:
        results = await asyncio.gather(
            chat(s, url, [{"role": "user", "content":
                 long_ctx + "\n\nSummarize the passage above in one sentence."}],
                 80, 0.7, "C1-long"),
            chat(s, url, [{"role": "user", "content":
                 "What is 17 * 23? Show the multiplication step by step."}],
                 120, 0.7, "C2-math"),
            chat(s, url, [{"role": "user", "content":
                 "Write a haiku about autumn."}], 60, 0.7, "C3-haiku"),
            chat(s, url, [{"role": "user", "content":
                 "Explain in two sentences why the sky appears blue."}],
                 80, 0.0, "C4-sky-greedy"),
        )
    return results


async def phase_d(url, n_tokens=400):
    """Steady-state aggregate throughput: 4 concurrent long generations on short
    prompts (prefill amortized away; rounds run at B=4 for most of the window)."""
    print(f"=== PHASE D: 4 concurrent x {n_tokens} tokens (aggregate throughput) ===",
          flush=True)
    stats = {}
    prompts = [
        "Write a detailed essay about the history of the printing press.",
        "Explain how a modern CPU executes an instruction, step by step.",
        "Write a long descriptive paragraph about a mountain storm.",
        "Describe the water cycle in detail, covering every stage.",
    ]
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1200)) as s:
        await asyncio.gather(*[
            chat(s, url, [{"role": "user", "content": p}], n_tokens, 0.7,
                 f"D{i + 1}", stats=stats)
            for i, p in enumerate(prompts)
        ])
    wall = time.perf_counter() - t0
    total_tok = sum(n for n, _ in stats.values())
    print(f"[D-aggregate] {total_tok} tokens / {wall:.1f}s wall = "
          f"{total_tok / wall:.1f} tok/s TOTAL across 4 streams "
          f"({total_tok / wall / 4:.1f} tok/s per stream)", flush=True)


async def phase_e(url):
    """Partial prefix reuse (same slot, divergent question) + cross-slot clone with
    the source slot free/busy. Server log must show 'partial reuse' for E2 and a
    'clone' line for E3/E4."""
    print("=== PHASE E: partial prefix reuse + shared-prefix clone ===", flush=True)
    shared = "Reference document:\n" + filler_tokens(9000) + "\n\n"
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)) as s:
        await chat(s, url, [{"role": "user", "content":
                 shared + "Question: what did engineers rely on instead of complete "
                 "understanding? Answer in one sentence."}], 60, 0.7, "E1-doc")
        await chat(s, url, [{"role": "user", "content":
                 shared + "Question: name two layers of abstraction mentioned in the "
                 "document. Answer in one sentence."}], 60, 0.7, "E2-partial")
        await asyncio.gather(
            chat(s, url, [{"role": "user", "content":
                 shared + "Question: what hid the complexity of the layer below? "
                 "Answer in one sentence."}], 60, 0.7, "E3-clone"),
            chat(s, url, [{"role": "user", "content":
                 shared + "Question: what did each new layer create? "
                 "Answer in one sentence."}], 60, 0.7, "E4-clone"),
        )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5001")
    ap.add_argument("--phases", default="ABC")
    ap.add_argument("--long-chars", type=int, default=60000,
                    help="phase C long-prompt size in chars (scale to the cache)")
    ap.add_argument("--d-tokens", type=int, default=400)
    args = ap.parse_args()

    # health check
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
        async with s.get(args.url + "/health") as r:
            if r.status != 200:
                raise SystemExit(f"server not healthy: {r.status}")

    a_texts = []
    if "A" in args.phases:
        a_texts = await phase_a(args.url)
    if "B" in args.phases:
        prior = a_texts[0] if a_texts else "The sea is vast and restless, grey under " \
            "a low sky, waves folding over one another without end."
        await phase_b(args.url, prior)
    if "C" in args.phases:
        await phase_c(args.url, args.long_chars)
    if "D" in args.phases:
        await phase_d(args.url, args.d_tokens)
    if "E" in args.phases:
        await phase_e(args.url)
    print("=== SMOKE CLIENT DONE (check server log for clone/hit/recapture lines) ===",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
