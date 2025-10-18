# Paper Quine Demo

Deterministic byte-level language model built from a tiny NumPy linear network that emits the LaTeX source of its own paper, demonstrating a practical take on the “paper quine” concept.

## Getting Started

```bash
make setup
source .venv/bin/activate
make all
```

The `verify.py` entrypoint runs the deterministic decode, writes `artifacts/paper.tex`, and emits the SHA-256 recorded in the manuscript.
