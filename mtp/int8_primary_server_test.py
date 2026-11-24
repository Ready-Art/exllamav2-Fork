"""Server-level INT8-primary KV test against a RUNNING tabbyAPI fork server.

Phases:
  S1 short chat (coherence)
  S2 ~25k-token filler prompt + 600 gen (long-ctx coherence + envelope recaptures)
  S3 continuation of S2 (prompt-cache hit -> INT8->fp16 dequant staging)
  S4 two concurrent requests (B=2; needs max_batch_size >= 2)

Usage: venv/bin/python mtp/int8_primary_server_test.py [--url http://127.0.0.1:5001] [--skip-b2]
FAIL = any request error, empty/degenerate completion.
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


def degenerate(text):
    return (text.count("la la") > 10 or text.count("inging") > 5
            or text.count(" the the") > 5 or text.count("C.") > 3)


async def chat(session, url, messages, max_tokens, tag, temperature=0.7):
    # Match the user's request shape: streaming, temp 0.7 / top_p 0.9 / top_k 64.
    payload = {"messages": messages, "max_tokens": max_tokens,
               "temperature": temperature, "top_p": 0.9, "top_k": 64,
               "stream": True}
    t0 = time.perf_counter()
    text = ""
    n = 0
    first = None
    async with session.post(url + "/v1/chat/completions", json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"[{tag}] HTTP {resp.status}: {(await resp.text())[:400]}")
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
                raise RuntimeError(f"[{tag}] stream error: {json.dumps(obj)[:400]}")
            piece = obj["choices"][0].get("delta", {}).get("content") or ""
            if piece:
                if first is None:
                    first = time.perf_counter() - t0
                text += piece
                n += 1
    dt = time.perf_counter() - t0
    tps = n / dt if dt > 0 else 0
    ok = n > 0 and not degenerate(text)
    print(f"[{tag}] {'OK' if ok else 'FAIL'} n={n} tps={tps:.1f} "
          f"first={first:.2f}s tail={text[-64:]!r}", flush=True)
    return ok, text, messages


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5001")
    ap.add_argument("--skip-b2", action="store_true")
    args = ap.parse_args()
    results = []
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1800)) as s:
        # S1 short
        ok, _, _ = await chat(s, args.url,
            [{"role": "user", "content": "Write a short paragraph about why the sky is blue."}],
            200, "S1-short")
        results.append(ok)

        # S2 ~25k tokens of filler (~4 chars/token -> 100k chars)
        filler = (FILLER * 400)[:100000]
        msgs2 = [{"role": "user", "content":
                  filler + "\n\nSummarize the passage above in three sentences."}]
        ok, text2, _ = await chat(s, args.url, msgs2, 4096, "S2-25k-4096gen")
        results.append(ok)

        # S3 continuation (prompt-cache hit)
        msgs3 = msgs2 + [{"role": "assistant", "content": text2},
                         {"role": "user", "content": "Now expand the second sentence into a paragraph."}]
        ok, _, _ = await chat(s, args.url, msgs3, 300, "S3-cont")
        results.append(ok)

        # S4 concurrent B=2
        if not args.skip_b2:
            async def one(i):
                return await chat(s, args.url,
                    [{"role": "user", "content":
                      f"Request {i}: explain in detail how a transformer attention head works."}],
                    400, f"S4-b2-{i}")
            r = await asyncio.gather(one(1), one(2), return_exceptions=True)
            for x in r:
                results.append(isinstance(x, tuple) and x[0])
    print("SERVER_TEST", "PASS" if all(results) else "FAIL", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
