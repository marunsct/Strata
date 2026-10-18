# Strata on the RX 7900 XTX (gfx1100): one card, and two cards on x16 + x4

The RX 7900 XTX / XT is `gfx1100` (RDNA3). It runs the Linux HIP backend ([AMD_HIP.md](AMD_HIP.md)); on Windows the
ready-made HIP engine. This page collects what is measured on this card in this repository, what the engine turns on
for other AMD cards but not (yet) for this one, and how to measure those on your PC before you keep them. Nothing here
changes a default start: every setting is opt-in, and the plain start stays byte-identical to the release.

## What a 7900 XTX gets today without doing anything

Setup writes these on its own:

- **hipBLASLt tuning table** for the prompt's dense matrix products (`STRATA_HIPBLASLT_TUNING`, `tools/hip/gfx1100-hipblaslt-<version>.txt`
  for the hipBLASLt version it installs; the engine refuses a table of another version and falls back to plain hipBLAS, about
  45% slower on a 131K prompt). Measured +40-60% prompt speed on the 7900 XTX ([AMD_HIP.md](AMD_HIP.md#tuning-table)).
- **HIP MMQ prompt experts** (`STRATA_PREFILL_MMQ`, on where the build has it): fresh-prompt prefill 1.8-2.0x over the earlier HIP
  runtime, 4K-9K prompts, one 7900 XTX ([AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md)).
- The per-card PCIe probe at start (`strata generate: PCIe probe: N GB/s ... -> pcie_frac`): the share of the missing experts each
  card reads over its own link. On a split every card is probed on its own slot, so an x4 card gets a smaller share than the x16 card
  without a setting ([MULTI_GPU.md](MULTI_GPU.md)).

## What the engine turns on for RDNA4 and Strix Halo, but not for gfx1100

`src/core/arch_defaults.cpp` switches a set of exact (bit-identical) faster kernels on by default on gfx12 (RX 9070 / R9700) and
gfx1151 (Strix Halo), and setup writes the same set for the Radeon 780M (gfx1103, an RDNA3 iGPU: same greedy answers with all 17 on
and off, +13-15% prompt, +3.2% decode). None of them is gated to an architecture in the kernels; none was measured by the
maintainers on a 7900 XTX. So for gfx1100 the table is **opt-in**:

| `STRATA_GFX1100_DEFAULTS` | turns on | measured where |
|---|---|---|
| `1` | the prompt half: `STRATA_GDN_HEAD=1 GDN_PP=2 GDN_CONVL2=1 GDN_NOY=1 CVEC_FUSE=1`, and `STRATA_SPLIT_OVERLAP=1` | the prompt half bitwise on gfx1201 (R9700), gfx1151 and the 780M; the overlap on **2x RX 7900 XTX** (below) |
| `2` | `1` plus the decode half: `Q8_PACKED Q6_PACKED MMVF_ROWS ATTN_LANECELL EXPERT_V2 TSUM LFUSE GDN_SPLIT QFUSE PLE_BATCH` | bitwise on gfx1151 and the 780M; on an R9700 the decode half measured **slower**, which is why gfx12 does not get it - measure, do not assume |

A switch you set yourself (in the config's `"env"` or the shell) is never overridden, and `STRATA_SH_STREAM` is left to setup's HIP
calibration. The engine prints one line at start naming what it set:
`strata: gfx1100 (RX 7900): 6 speed switches on by STRATA_GFX1100_DEFAULTS=1 ...`.

To keep it: run setup with the variable in its environment, and it is written into the config's `"env"`:

```sh
STRATA_GFX1100_DEFAULTS=1 ./setup.sh
```

A setup run without the variable takes it out again. Or add `"STRATA_GFX1100_DEFAULTS": "1"` to the `"env"` block of
`strata-<model>.json` and restart.

## Measure first: `tools/hip/bench_gfx1100.py`

```sh
.venv/bin/python tools/hip/bench_gfx1100.py              # the newest strata-*.json; 3 rounds, server restarted per arm
.venv/bin/python tools/hip/bench_gfx1100.py --list       # which arms this config gets
.venv/bin/python tools/hip/bench_gfx1100.py --arms overlap,pw2 --rounds 5
```

It runs `tools/ab_engine.py` with the arms that fit the config (one card or a split), alternating the order every round, and prints
the per-arm medians of decode tok/s (three short chats) and prompt tok/s (one ~8K-token prompt). Each arm's greedy text is in
`bench_gfx1100.jsonl`: an arm whose text differs from `base` changes the output bits; the arms below say which ones can.
Compare medians over rounds, on an idle PC, and keep an arm only when it wins in every round and by more than the spread of
`base` against itself.

## One RX 7900 XTX (24 GB)

Arms, and what is known:

- `p1` / `p2`: the table above. Expected direction from the 780M and gfx1151: prompts a few percent to +15% from `p1`; decode is the
  open question (`p2`).
- `bits`: `STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_PF_SWITCH_MIN_T=4096`. A reporter on a 7900 XTX (50 GB RAM, `--resident-experts`)
  read a 74K prompt at 925 -> 1,092 tok/s with them ([AMD_HIP.md](AMD_HIP.md), #1389). **Not bit-identical** (rounding-level), so
  read the text column.
- `gdnc`: `STRATA_GDN_CHUNKED=1`, the prompt's DeltaNet recurrence in 16-token chunks on the gfx11 matrix cores
  ([AMD_HIP.md](AMD_HIP.md#opt-in-chunked-prompt-gdn-on-gfx11)): 1.6x less time per layer on gfx1151, built for gfx1100 but not run
  there yet. **Not bit-identical** (FP16 operands).
- Long prompts (from ~65K tokens), not in the script: `STRATA_SCORES_THREAD=1`, the exact long-prompt scorer; on one 7900 XTX a fresh
  131K-token request was 10-13% faster and a 240K one 22% ([AMD_HIP.md](AMD_HIP.md#exact-long-prompt-qsa-scorer-gfx1100-opt-in-gfx12-default)),
  same tokens. Nothing below 65K.
- Already in the README for any one-card PC whose RAM holds every expert: `"--adapt-async", "1"` (decode +4-5% on an RTX 5070; AMD
  not measured) and, with `--resident-experts`, check the `lendable slots keep their experts in RAM too` line after the first request
  ([AMD_HIP.md](AMD_HIP.md)): when N of M is short, lower `STRATA_RESIDENT_HEADROOM_GIB` by one and restart.
- Not worth trying on this card (measured flat by the same reporter): `--kv-resident` changes, a prefill chunk above 32768, `--spec` above 4.

## Two RX 7900 XTX, one on PCIe 4.0 x16 and one on x4

Setup lists both cards; `./setup.sh --backend hip --gpus 0,1` (or the menu's "together" entry) runs the layer split: layers 0..K-1 on
the first card, K..47 on the second, which also runs the head and the draft layer ([MULTI_GPU.md](MULTI_GPU.md)). A token crosses the
cards once per verify window, a few hundred KB through pinned RAM, so the x4 link does not slow the hand-off; what it slows is the
**experts that card streams** for a window (its misses). What to know:

1. **Card order.** Two identical cards keep the order you give. Put the **x16 card first** (`--gpus <x16>,<x4>`): the first card runs
   the prompt path's first stage (it streams the experts of every chunk its cache does not hold) and, with `--pipeline-windows`, the
   draft layer. Check with `nvidia-smi`'s AMD counterpart: `cat /sys/class/drm/card*/device/current_link_width` says `16` and `4`.
2. **Where the layers land (`"layer_split": "auto"`).** Auto prices each card's misses by its probed link (from a 4x gap on, #2007),
   which on a 2x RTX 3090 x16 + x4 rig moved the split to K=31 and cost 10-15% of decode against the balanced K=27 (#2418). Two arms
   measure it here: `uniform` (`STRATA_SPLIT_UNIFORM=1`, every link priced the same) and a manual sweep of `"layer_split"`
   (`"24"`, `"26"`, `"28"`; the engine log line `layer split: ... chose` says what auto picked). Prefill that is flat from 4K to 16K
   tokens is the sign of an idle card.
3. **The hand-off (`overlap`, in `p1`).** `STRATA_SPLIT_OVERLAP=1` was measured on exactly this pair (2x RX 7900 XTX, Ryzen 9 9900X, Swift 1.5
   IQ3_XXS, 128K context, int8 KV, layers 0-25 / 26-47, median of 8): decode 53 -> 70.1 tok/s on a code prompt and 45 -> 55.9 on prose,
   byte-identical output ([MULTI_GPU.md](MULTI_GPU.md#overlapped-hand-off-strata_split_overlap1-opt-in)). One card alone was ~70 / ~68, so with the
   overlap the pair decodes like one card and reads prompts faster.
4. **The pipelined loop (`pw2`).** `--pipeline-windows 2` is the default on gfx12 splits and measured +14-27% decode on 4x RX 7900 XT
   (gfx1100, every expert in VRAM). It and the overlap are two ways to hide the same wait; the engine turns the overlap off beside the
   pipeline and says so. Which wins on two 7900 XTX is not measured: that is the `overlap` against `pw2` comparison.
5. **More experts per card (`trim`).** `STRATA_STAGE_TRIM=1` loads each card only its layers' dense weights (~3.4 GB freed per card for the
   Coder; 2x MI50: 8,819 -> 10,626 experts in VRAM, decode +6%). It needs a numeric `"layer_split"`, and a card that holds more experts
   can route a few more on the GPU, so the text can differ slightly.
6. **The display.** If one card drives the monitors, give the other card the room: `--vram-reserve-mib 300 --vram-reserve-later-mib 1800`
   puts the headroom on the second card when the display is there ([MULTI_GPU.md](MULTI_GPU.md)).
7. **Not for this pair:** `STRATA_PREFILL_HELP=1` (the idle card helps one-chunk prompts) is measured on 2x RTX 3090 only and streams
   over the helper's own link - on an x4 card that is the slow link. Try it last, and only on short prompts.

A reasonable order: `bench_gfx1100.py --arms overlap,pw2`, keep the winner; then `--arms uniform` against it with the
`"layer_split"` sweep; then `p2`. Keep the engine log of every arm (`/tmp/ab_<arm>_r<round>.log`): the `layer split`, `PCIe probe`
and `speed switches on` lines say what actually ran.

## Reporting

A report from this hardware is welcome: [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md), with the `bench_gfx1100.jsonl`, the
arms' engine logs, ROCm version, the two cards' link widths and the config. Say what you did not test.
