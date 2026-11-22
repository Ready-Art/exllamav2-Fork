"""Server-level smoke test for the prompt-cache RAM offload tier (tabbyAPI fork).

Drives a RUNNING tabbyAPI fork server started with:
    model.prompt_cache_offload: true
    model.prompt_cache_ram_limit_mb: <cap>
    model.max_batch_size: <B>      (mp backend + MTP drafter)

Phases:
  O1  alternating conversations (> B distinct lineages, 2 turns each). Turn 2 of
      each conversation must reuse its lineage -- hot (prompt cache hit) or COLD
      (prompt cache cold restore) -- never a full re-prefill of the document.
  O2  RAM-cap eviction: many distinct long lineages under a small cap; the store
      must evict oldest/least-used ("evict cold lineage ... RAM cap") and never
      crash or OOM.
  O3  concurrency: 4 simultaneous streams + continuations with offload enabled
      (clone / restore / offload interplay under load).
  O4  re-rendered history from a COLD lineage: turn 2 edits the assistant turn,
      forcing the prompt-boundary restore / partial path after a cold restore.

PASS = every completion non-empty + coherent, no HTTP/stream errors. The SERVER
log must show "[prompt-offload] ... lineage -> RAM", "cold restore", and (O2)
"evict cold lineage"; any "[fi-guard] CORRUPTION", "[TP graphs] disabled",
traceback, or "cold restore ... failed" = FAIL.

Usage:
  venv/bin/python exllamav2-Fork/mtp/offload_smoke_client.py --url http://127.0.0.1:5001 --phases O1O2O3O4
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
               min_tokens=0):
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    if min_tokens > 0:
        payload["min_tokens"] = min_tokens
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
                raise RuntimeError(f"[{tag}] server stream error: {json.dumps(obj)[:800]}")
            delta = obj["choices"][0].get("delta", {})
            piece = delta.get("content") or ""
            if piece:
                if first_tok is None:
                    first_tok = time.perf_counter() - t0
                text += piece
                n_chunks += 1
    dt = time.perf_counter() - t0
    print(f"[{tag}] {dt:.1}s total, first-chunk {first_tok if first_tok else -1:.1}s, "
          f"{n_chunks} tok, {n_chunks / max(dt, 1e-9):.1f} tok/s", flush=True)
    if not text.strip():
        raise RuntimeError(f"[{tag}] EMPTY completion")
    print(f"[{tag}] head: {text[:120]!r}", flush=True)
    return text


def doc_question(i, chars):
    doc = f"Reference document {i}:\n" + filler_tokens(chars) + "\n\n"
    q = ("Question: according to the document, what did engineers rely on instead "
         "of complete understanding? Answer in one sentence.")
    return doc, q


async def phase_o1(url, n_convs=4, chars=6000):
    print(f"=== PHASE O1: {n_convs} alternating conversations x 2 turns "
          f"(cold restore expected) ===", flush=True)
    docs = [doc_question(i, chars) for i in range(n_convs)]
    answers = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as s:
        # turn 1 for every conversation (fills VRAM rows, evicts the coldest to RAM)
        for i, (doc, q) in enumerate(docs):
            answers[i] = await chat(
                s, url, [{"role": "user", "content": doc + q}], 60, 0.7, f"O1-T1-c{i}")
        # turn 2 for every conversation: strict extension of its own lineage. With
        # n_convs > B the early conversations were evicted -> COLD restore.
        for i, (doc, q) in enumerate(docs):
            msgs = [
                {"role": "user", "content": doc + q},
                {"role": "assistant", "content": answers[i]},
                {"role": "user", "content":
                 "Now restate that answer in exactly five words."},
            ]
            await chat(s, url, msgs, 40, 0.7, f"O1-T2-c{i}")


async def phase_o2(url, n_convs=8, chars=12000):
    print(f"=== PHASE O2: {n_convs} distinct long lineages (RAM-cap eviction) ===",
          flush=True)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as s:
        for i in range(n_convs):
            doc, q = doc_question(100 + i, chars)
            await chat(s, url, [{"role": "user", "content": doc + q}], 40, 0.7,
                       f"O2-c{i}")


async def phase_o3(url, n_tokens=300):
    print(f"=== PHASE O3: 4 concurrent x {n_tokens} tok + continuations ===", flush=True)
    prompts = [
        "Write a detailed essay about the history of the printing press.",
        "Explain how a modern CPU executes an instruction, step by step.",
        "Write a long descriptive paragraph about a mountain storm.",
        "Describe the water cycle in detail, covering every stage.",
    ]
    stats_texts = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as s:
        texts = await asyncio.gather(*[
            chat(s, url, [{"role": "user", "content": p}], n_tokens, 0.7, f"O3-{i}")
            for i, p in enumerate(prompts)
        ])
        # two continuations (own-lineage hot reuse or cold restore)
        for i in (0, 2):
            await chat(s, url, [
                {"role": "user", "content": prompts[i]},
                {"role": "assistant", "content": texts[i]},
                {"role": "user", "content": "Summarize your answer in one sentence."},
            ], 60, 0.7, f"O3-cont{i}")


async def phase_o4(url, chars=8000):
    print("=== PHASE O4: re-rendered history from a cold lineage ===", flush=True)
    doc, q = doc_question(7, chars)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as s:
        ans = await chat(s, url, [{"role": "user", "content": doc + q}], 120, 0.7,
                         "O4-T1")
        # push the lineage to RAM with other conversations
        for i in range(3):
            d2, q2 = doc_question(200 + i, 4000)
            await chat(s, url, [{"role": "user", "content": d2 + q2}], 30, 0.7,
                       f"O4-evict{i}")
        # turn 2 re-renders the assistant turn (truncated) -> divergence inside the
        # generated history; the cold restore + boundary-restore/partial path handles it
        rerendered = ans[: max(1, len(ans) // 2)] + " [edited]"
        await chat(s, url, [
            {"role": "user", "content": doc + q},
            {"role": "assistant", "content": rerendered},
            {"role": "user", "content": "Continue from that edited answer."},
        ], 60, 0.7, "O4-T2-rerendered")


async def phase_o5(url, gen_tokens=3200, chars=6000):
    """Long generation shifts the SWA ring PAST the prompt boundary; a thought-stripped
    continuation of the (now cold) lineage must rewind via the attached boundary
    snapshot -- prefilling from the boundary, NOT re-prefilling the document. Server
    log must show 'cold restore + boundary rewind' with 'prefilling <tail> new' where
    the start is the prompt boundary (document cached), not 0."""
    print(f"=== PHASE O5: long generation ({gen_tokens} tok) + cold thought-strip "
          f"continuation (snapshot rewind) ===", flush=True)
    doc, _ = doc_question(55, chars)
    # Force a generation longer than the SWA ring depth so window_beg advances past
    # the prompt boundary -- only the attached boundary snapshot can rewind it exactly.
    long_q = ("Write an extremely long, detailed, multi-section essay about the history "
              "of computing, covering mechanical calculators, vacuum tubes, transistors, "
              "integrated circuits, personal computers, the internet, and modern AI. "
              "Do not stop early; keep writing thorough detail for many paragraphs.")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as s:
        ans = await chat(s, url, [{"role": "user", "content": doc + long_q}],
                         gen_tokens, 0.7, "O5-long-T1", min_tokens=gen_tokens - 50)
        # evict the long lineage to RAM with other conversations
        for i in range(3):
            d2, q2 = doc_question(300 + i, 4000)
            await chat(s, url, [{"role": "user", "content": d2 + q2}], 30, 0.7,
                       f"O5-evict{i}")
        # thought-stripped continuation (content-only assistant) of the cold lineage
        await chat(s, url, [
            {"role": "user", "content": doc + long_q},
            {"role": "assistant", "content": ans},
            {"role": "user", "content": "Summarize that in one sentence."},
        ], 40, 0.7, "O5-cold-continuation")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5001")
    ap.add_argument("--phases", default="O1O2O3O4O5")
    ap.add_argument("--convs", type=int, default=4)
    args = ap.parse_args()

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
        async with s.get(args.url + "/health") as r:
            if r.status != 200:
                raise SystemExit(f"server not healthy: {r.status}")

    if "O1" in args.phases:
        await phase_o1(args.url, args.convs)
    if "O2" in args.phases:
        await phase_o2(args.url)
    if "O3" in args.phases:
        await phase_o3(args.url)
    if "O4" in args.phases:
        await phase_o4(args.url)
    if "O5" in args.phases:
        await phase_o5(args.url)
    print("=== OFFLOAD SMOKE DONE (grep server log for [prompt-offload] / cold restore "
          "/ evict; any CORRUPTION / graphs disabled / traceback = FAIL) ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
