# Paper Quine Demo

Deterministic byte-level language model built from a tiny NumPy linear network that now emits a self-reconstructing Python script for the entire repository—including the weight archive that powers the decode—fulfilling the “true AI quine” goal.

## Getting Started

```bash
make setup
source .venv/bin/activate
make all
```

The `verify.py` entrypoint runs the deterministic decode, writes `artifacts/paper.tex`, and emits the SHA-256 recorded alongside the bundled weights.

### Regenerating the artifact

Whenever source files change, refresh the emitted script and weight bundle via:

```bash
python scripts/build_true_quine.py
```

The command rewrites `src/paper_quine/paper.tex`, the corresponding SHA-256, and the weight payload so that the model’s decode matches the new repository snapshot.
