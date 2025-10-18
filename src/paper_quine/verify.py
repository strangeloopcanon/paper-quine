"""Verification CLI for the paper quine project."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from .model import decode_paper

DEFAULT_OUTPUT = Path("artifacts/paper.tex")
REFERENCE_LATEX = Path("src/paper_quine/paper.tex")
PAPER_DIGEST_PATH = Path("src/paper_quine/data/paper.sha256")


def run_llm_live(output_path: Path) -> int:
    """Execute the deterministic decode pipeline and validate all hashes."""

    decoded = decode_paper()
    expected = REFERENCE_LATEX.read_bytes()

    if decoded != expected:
        print("Decoded bytes diverge from repository source.")
        return 1

    digest = hashlib.sha256(decoded).hexdigest()
    recorded_digest = PAPER_DIGEST_PATH.read_text(encoding="utf-8").strip()

    if digest != recorded_digest:
        print(
            "Digest mismatch:\n"
            f"  expected: {recorded_digest}\n"
            f"  actual:   {digest}"
        )
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(decoded)

    print("Deterministic decode successful.")
    print(f"sha256(paper.tex) = {digest}")
    print(f"output written to {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paper quine verification tooling.")
    parser.add_argument(
        "--llm-live",
        action="store_true",
        help="Run deterministic LLM live tests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write the decoded paper (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args(argv)

    if args.llm_live:
        return run_llm_live(args.output)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
