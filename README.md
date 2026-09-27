# Tsui's Leftover Freedom — research artifact and its measurements

Which parts of Chia-Chi Tsui's robust-control machinery (advanced state-space
techniques, observable Hessenberg form) actually transfer to transformer
fine-tuning — LoRA, SVD adapters, and closed-form model editing.

**Published artifact (Version 3):** https://claude.ai/artifact/WbEhq2XT6kXzxHYNAmjQBZ

## Provenance

Researched and written 2026-09-26/27 in a Claude Code session whose working
directory was `_work/a2a-xpr`. It is *not* part of that project — a2a-xpr is an
agent-stack docs repo and nothing here belongs to it. These files lived in an
ephemeral session scratchpad under `/private/tmp/claude-502/...` and were moved
here on 2026-09-27 so a tmp cleanup could not take them.

## Layout — deliberately flat

Every script reads **bare filenames out of its own cwd** (`open("corpus.txt")`,
`open('measure.json')`, `open('tsui-fine-tuning.html')`). Sorting them into
`measure/` and `results/` subdirectories would break all of them, so the flat
layout is load-bearing, not laziness. Only `screenshots/` is split out, because
nothing reads it.

| File | Role |
| --- | --- |
| `tsui-fine-tuning.html` | **Source of truth for the page.** Republish with the Artifact tool using this path, or pass the URL above as `url=`. |
| `check.js` | The page's script block, extracted for linting. Derived from the HTML — edit the HTML, not this. |
| `measure.py` · `measure_xl.py` · `measure_gptj.py` | The leakage measurement, per model. |
| `spectrum.py` | Layer-6 key-covariance spectrum for the chart. |
| `merge_measured.py` | Folds the JSON results into `measured.js` for the page. |
| `apply_xl.py` | Writes measured numbers into `tsui-fine-tuning.html` in place. |
| `probe.js` … `probe4.js` | Small numerical probes of the maths claims (e.g. whether the least-fragile point on the freedom line differs from the minimum-norm point). |
| `corpus.txt` | 1.98 MB / ~496k tokens of Wikipedia plaintext, grown across runs. |
| `run_gptj.sh` | Wrapper for the GPT-J run. Rewritten on the move — the original hardcoded two now-dead scratchpad paths and waited on background task files from that session. |
| `screenshots/` | Verification screenshots of the rendered page. |

**Never hand-type a measured number into the page** — generate it from the JSON
via `merge_measured.py` / `apply_xl.py`.

## Measurements COMPLETE (results already in the published page)

- `measure.json` — GPT-2 small (124M), layers 3/6/9, key dim 3072
- `measure_gpt2-xl.json` — GPT-2 XL (1.5B), layers 14/17/36, key dim 6400
- `spectrum6.json` — full layer-6 spectrum for the chart
- `measure_page.json`, `measured.js` — what the page actually consumes

Headline: MEMIT's `K₀K₀ᵀ` covariance term is worth **7.9–11.4×** (small) /
**6.6–11.9×** (XL) in RMS leakage onto held-out keys at identical edit accuracy.
Participation ratio 17.5–27.5 of 3072 and 30.8–66.9 of 6400 → 0.57–0.90% vs
0.48–1.05% of directions. Scale-invariant across a 12× parameter gap.

## Measurement FAILED — GPT-J 6B

`measure_gptj.py` started and was killed with no traceback ⇒ out of memory.
Budget was ~20 GB (12 GB fp16 weights + 6 GB of three 16384² float64 Gramians +
2.2 GB keys) on a 36 GB machine. Two likely contributors:

1. the `except` fallback can load the 23 GB fp32 checkpoint — both revisions had
   landed in the cache, which is how we know it fired. Remove the fallback and
   pin `revision="float16"` so a retry cannot silently pull fp32;
2. three simultaneous 16384² float64 Gramians. Fix: one layer per sweep, or
   accumulate in float32 (3 GB), or drop to 2 layers.

Nothing about GPT-J is in the published page, so its absence breaks nothing.
Worth weighing before spending ~90 minutes on a retry: **GPT-2 XL already
answered the scaling question, and it is the model MEMIT itself was benchmarked
on.** GPT-J would be a third point, not a missing one.

## Disk note

The GPT-J cache (34 GB: 12 GB fp16 + 23 GB fp32 blobs) was **deleted 2026-09-27**
at the user's request — disk went 97 GB → 131 GB free. A GPT-J retry therefore
re-downloads from scratch (~12 min for fp16 alone). The GPT-2 and GPT-2 XL caches
are kept (526 MB + 6.0 GB) because the published measurements cite them.

## Environment

`requirements.txt` pins what the original runs used: python 3.9, torch 2.8.0,
transformers 4.57.6, numpy 2.0.2. MPS available on this machine. The 494 MB
`.venv` was **not** copied here — recreate it:

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

`corpus.txt` grew across runs, so samples-per-key-dimension differed per model
(30.3 / 26.9 / 17.1) — all three figures are stated on the page.

## Known caveats already written into the page

Synthetic edit targets (one real key made to emit another's value, not a
ROME-optimised v\*); leakage is a proxy for downstream damage; corpora differ per
model. All three are stated in section 06 at depth 3.
