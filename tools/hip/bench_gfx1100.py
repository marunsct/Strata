#!/usr/bin/env python3
"""A/B the RX 7900 XTX / XT (gfx1100) settings on this PC, one or two cards (docs/RX7900XTX.md).

    .venv/bin/python tools/hip/bench_gfx1100.py                    # the newest strata-*.json of this folder
    .venv/bin/python tools/hip/bench_gfx1100.py strata-iq3_xxs.json --rounds 3
    .venv/bin/python tools/hip/bench_gfx1100.py --list             # print the arms and stop

It hands tools/ab_engine.py the arms that fit the config - the default start first, then each opt-in - and ab_engine
restarts the server per arm, alternates the order over rounds and prints the per-arm medians (decode tok/s on three
short chats, prompt tok/s on one ~8K-token prompt).  Every arm's greedy text is in ab_engine.jsonl: an arm whose text
differs from "base" changes the output bits, which the doc says beside each one.

One card (no "gpu" list in the config):
  base        the config as it is
  p1          STRATA_GFX1100_DEFAULTS=1   the prompt half of the gfx12 table (exact there and on the 780M)
  p2          STRATA_GFX1100_DEFAULTS=2   plus the gfx1151 decode half
  bits        STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_PF_SWITCH_MIN_T=4096 - a reporter's 7900 XTX prompt levers;
              NOT bit-identical (rounding), so judge the text too
  gdnc        STRATA_GDN_CHUNKED=1        the prompt's DeltaNet recurrence in 16-token chunks on the gfx11 matrix cores
              (1.6x less time per layer on gfx1151; not run on gfx1100 yet); NOT bit-identical
Two or more cards (a layer split):
  base        the config as it is (serial hand-off)
  overlap     STRATA_SPLIT_OVERLAP=1      measured on 2x RX 7900 XTX: decode +32% code / +24% prose, same bytes
  p1 / p2     as above (p1 includes the overlap)
  pw2         --pipeline-windows 2 + STRATA_MTP_STAGE0=1   the gfx12 default loop with the draft layer on the first card
              (without that, two stages lost to the serial loop on 2x R9700; needs the pack's draft_vocab.bin); the engine
              turns the overlap off beside it
  uniform     STRATA_SPLIT_UNIFORM=1      every link priced the same under "auto" (for an x16 + x4 pair)
  trim        STRATA_STAGE_TRIM=1         each card loads only its layers' dense weights - only with a numeric
              "layer_split" in the config; a card then holds more experts, so the text can differ

--arms NAME,NAME limits the run to those arms (base is always first).  Standard library only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ONE = [
    ("p1", "STRATA_GFX1100_DEFAULTS=1"),
    ("p2", "STRATA_GFX1100_DEFAULTS=2"),
    ("bits", "STRATA_PF_FUSED=1,STRATA_PF_GEMM=1,STRATA_PF_SWITCH_MIN_T=4096"),
    ("gdnc", "STRATA_GDN_CHUNKED=1"),
]
SPLIT = [
    ("overlap", "STRATA_SPLIT_OVERLAP=1"),
    ("p1", "STRATA_GFX1100_DEFAULTS=1"),
    ("p2", "STRATA_GFX1100_DEFAULTS=2"),
    ("pw2", "STRATA_MTP_STAGE0=1,xargs=--pipeline-windows;2"),
    ("uniform", "STRATA_SPLIT_UNIFORM=1"),
    ("trim", "STRATA_STAGE_TRIM=1"),
]


def newest_config() -> Path | None:
    cfgs = sorted(ROOT.glob("strata-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cfgs[0] if cfgs else None


def arms_for(cfg: dict) -> list[tuple[str, str]]:
    gpus = cfg.get("gpu")
    split = isinstance(gpus, list) and len(gpus) > 1
    if not split:
        return list(ONE)
    arms = list(SPLIT)
    ls = str(cfg.get("layer_split", "auto"))
    if ls == "auto" or not ls.replace(",", "").isdigit():
        arms = [a for a in arms if a[0] != "trim"]   # STRATA_STAGE_TRIM needs explicit split points
    return arms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="strata-<model>.json (default: the newest in the Strata folder)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--arms", help="comma-separated arm names to run (base is always run)")
    ap.add_argument("--list", action="store_true", help="print the arms and stop")
    ap.add_argument("--log", default="bench_gfx1100.jsonl")
    a = ap.parse_args()
    cpath = Path(a.config) if a.config else newest_config()
    if cpath is None or not cpath.is_file():
        print("no strata-*.json found: run ./setup.sh first, or give the config path", file=sys.stderr)
        return 2
    cfg = json.loads(cpath.read_text(encoding="utf-8"))
    if cfg.get("backend") != "hip":
        print(f"{cpath.name} is not a HIP (AMD) config", file=sys.stderr)
        return 2
    arms = arms_for(cfg)
    if a.arms:
        want = {x.strip() for x in a.arms.split(",")} - {"base"}
        unknown = want - {n for n, _ in arms}
        if unknown:
            print(f"unknown arm(s) {sorted(unknown)}; this config offers {[n for n, _ in arms]}", file=sys.stderr)
            return 2
        arms = [x for x in arms if x[0] in want]
    env = cfg.get("env") or {}
    if "STRATA_GFX1100_DEFAULTS" in env:
        print(f"note: {cpath.name} already carries STRATA_GFX1100_DEFAULTS={env['STRATA_GFX1100_DEFAULTS']}: 'base' here is "
              "that profile, not the plain default (run setup again without the variable for a plain base)")
    spec = ["base:"] + [f"{n}:{v}" for n, v in arms]
    gpus = cfg.get("gpu")
    print(f"config {cpath.name}; GPUs {gpus if gpus is not None else 'one'}; layer split {cfg.get('layer_split', '-')}")
    for s in spec:
        print("  " + s)
    if a.list:
        return 0
    cmd = [sys.executable, str(ROOT / "tools" / "ab_engine.py"), str(cpath), *spec, "--rounds", str(a.rounds),
           "--port", str(a.port), "--log", a.log]
    print("running: " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    sys.exit(main())
