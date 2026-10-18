"""Tests for the opt-in RX 7900 XTX / XT (gfx1100) speed profile (docs/RX7900XTX.md): setup reads
STRATA_GFX1100_DEFAULTS=1|2 from its environment and writes it into the config, the engine's arch table
(src/core/arch_defaults.cpp) holds what the docs say, and nothing is written when the variable is unset.
No GPU, no ROCm, no downloads.

    python -m unittest tools.test_setup_gfx1100
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

PROMPT = {"STRATA_GDN_HEAD": "1", "STRATA_GDN_PP": "2", "STRATA_GDN_CONVL2": "1", "STRATA_GDN_NOY": "1",
          "STRATA_CVEC_FUSE": "1", "STRATA_SPLIT_OVERLAP": "1"}


def _engine_table():
    """The gfx1100 block of src/core/arch_defaults.cpp: (tier 1 switches, tier 2 additions)."""
    src = (ROOT / "src/core/arch_defaults.cpp").read_text(encoding="utf-8")
    block = src.split("gfx_arch_is_gfx1100(gcn_arch)) {", 1)[1].split("gfx_arch_is_gfx12(gcn_arch)) {", 1)[0]
    t1 = dict(re.findall(r'\{"(STRATA_[A-Z0-9_]+)",\s*"([^"]+)"\}', block.split("t = {", 1)[1].split("};", 1)[0]))
    t2 = dict(re.findall(r'\{"(STRATA_[A-Z0-9_]+)",\s*"([^"]+)"\}', block.split("if (tier >= 2)", 1)[1]))
    return t1, t2


class EngineTable(unittest.TestCase):
    def test_tier_one_is_the_prompt_half_plus_split_overlap(self):
        t1, t2 = _engine_table()
        self.assertEqual(t1, PROMPT)
        self.assertTrue(set(t2) & {"STRATA_Q8_PACKED", "STRATA_EXPERT_V2"})
        for k in ("STRATA_SH_STREAM", "STRATA_HCD_EXACT", "STRATA_PF_GEMM", "STRATA_PF_FUSED", "STRATA_HIP_WMMA"):
            self.assertNotIn(k, t1)
            self.assertNotIn(k, t2)

    def test_tier_two_is_the_gfx1103_decode_half(self):
        """The decode switches the 780M (gfx1103, RDNA3 too) runs exact, less STRATA_SH_STREAM (setup's calibration)."""
        _, t2 = _engine_table()
        decode = {k: v for k, v in setup.GFX1103_SWITCHES.items()
                  if k not in PROMPT and k not in ("STRATA_PF_PAD", "STRATA_SH_STREAM")}
        self.assertEqual(t2, decode)


class SetupReadsTheVariable(unittest.TestCase):
    def _reload(self, value):
        env = dict(os.environ)
        env.pop("STRATA_GFX1100_DEFAULTS", None)
        if value is not None:
            env["STRATA_GFX1100_DEFAULTS"] = value
        with mock.patch.dict(os.environ, env, clear=True):
            return importlib.reload(setup).GFX1100_PROFILE

    def tearDown(self):
        self._reload(None)

    def test_values(self):
        self.assertEqual(self._reload(None), "")
        self.assertEqual(self._reload("0"), "")
        self.assertEqual(self._reload(""), "")
        self.assertEqual(self._reload("yes"), "")
        self.assertEqual(self._reload("1"), "1")
        self.assertEqual(self._reload(" 2 "), "2")

    def test_is_setups_own_env_entry(self):
        self.assertIn("STRATA_GFX1100_DEFAULTS", setup.SETUP_ENV)   # a setup run without it takes it out again

    def test_only_exactly_gfx1100(self):
        self.assertTrue(setup.gfx_arch_is("gfx1100", "gfx1100"))
        self.assertTrue(setup.gfx_arch_is("gfx1100:sramecc-:xnack-", "gfx1100"))
        for other in ("gfx1101", "gfx1102", "gfx1103", "gfx11000", "gfx1201"):
            self.assertFalse(setup.gfx_arch_is(other, "gfx1100"))


class HipEnvIsMerged(unittest.TestCase):
    def test_the_hipblaslt_table_no_longer_replaces_the_unsloth_entries(self):
        """setup.py once did cfg["env"] = {"STRATA_HIPBLASLT_TUNING": ...} on HIP, which dropped STRATA_MMQ_KQ /
        STRATA_ROUTE_TAIL_SKIP written for Unsloth's files a few lines earlier (every gfx1100 install has a table)."""
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertNotIn('cfg["env"] = {"STRATA_HIPBLASLT_TUNING"', src)
        self.assertIn('cfg.setdefault("env", {})["STRATA_HIPBLASLT_TUNING"]', src)
        self.assertEqual(setup.unsloth_default_env("unsloth", "UD-Q6_K_XL", True, False),
                         {"STRATA_MMQ_KQ": "1", "STRATA_ROUTE_TAIL_SKIP": "0"})


if __name__ == "__main__":
    unittest.main()
