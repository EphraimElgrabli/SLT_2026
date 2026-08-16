from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "data" / "scripts" / "slt_data.py"
SPEC = importlib.util.spec_from_file_location("slt_data", MODULE_PATH)
assert SPEC and SPEC.loader
slt_data = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = slt_data
SPEC.loader.exec_module(slt_data)


class ArgumentForwardingTests(unittest.TestCase):
    def test_flag_args_uses_cli_spelling(self) -> None:
        args = argparse.Namespace(force=True, skip_drives=False, median_align=True)

        self.assertEqual(
            slt_data.flag_args(args, "force", "skip_drives", "median_align"),
            ["--force", "--median-align"],
        )

    def test_optional_value_args_preserves_zero(self) -> None:
        args = argparse.Namespace(limit=0, input_size=None)

        self.assertEqual(
            slt_data.optional_value_args(args, "limit", "input_size"),
            ["--limit", "0"],
        )

    def test_optional_list_args_skips_empty_values(self) -> None:
        args = argparse.Namespace(models=["vits", "vitb"], datasets=None)

        self.assertEqual(
            slt_data.optional_list_args(args, "models", "datasets"),
            ["--models", "vits", "vitb"],
        )


if __name__ == "__main__":
    unittest.main()
