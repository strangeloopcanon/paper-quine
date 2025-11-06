#!/usr/bin/env python3
"""Build the deterministic AI quine artifact and bundled weights.

This script constructs a self-extracting Python artifact whose source code is the
exact output emitted by the deterministic language model in
``paper_quine.model``. Running this tool regenerates the following derived
assets in the repository:

* ``src/paper_quine/paper.tex`` (the self-reconstructing artifact script)
* ``src/paper_quine/data/paper.sha256`` (digest of the artifact bytes)
* ``src/paper_quine/data/weights.b85`` (base85-encoded model payload)
* ``src/paper_quine/data/weights.sha256`` (checksum of the weight archive)

The generated artifact script knows how to rebuild the entire repository tree,
including recomputing the weight archive directly from its own bytes, making
the decode a "true" quine in the AI sense.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import subprocess  # nosec B404
import tarfile
import textwrap
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = Path("src/paper_quine/paper.tex")
DIGEST_PATH = Path("src/paper_quine/data/paper.sha256")
WEIGHTS_PATH = Path("src/paper_quine/data/weights.b85")
WEIGHTS_DIGEST_PATH = Path("src/paper_quine/data/weights.sha256")

GENERATED_PATHS = {
    ARTIFACT_PATH,
    DIGEST_PATH,
    WEIGHTS_PATH,
    WEIGHTS_DIGEST_PATH,
}

START_TOKEN = "<START>"  # nosec B105 - public sentinel bytes
END_TOKEN = "<END>"  # nosec B105 - public sentinel bytes
JSON_SEPARATORS = (",", ":")


def _wrap_lines(value: str, width: int = 80) -> str:
    """Wrap a long string to the requested column width."""

    return "\n".join(textwrap.wrap(value, width))


def _git_tracked_files() -> list[Path]:
    """Return the set of tracked files relative to the repository root."""

    output = subprocess.check_output(  # nosec B603 B607
        ["git", "ls-files"], cwd=REPO_ROOT, text=True
    )
    return [Path(line) for line in output.splitlines() if line.strip()]


def _build_archive(encoded_paths: list[Path]) -> tuple[str, str]:
    """Return archive payload (base85 text, sha256 digest of compressed bytes)."""

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for rel_path in sorted(encoded_paths):
            tar.add((REPO_ROOT / rel_path).resolve(), arcname=str(rel_path))
    tar_bytes = buffer.getvalue()
    compressed = zlib.compress(tar_bytes, level=9)
    digest = hashlib.sha256(compressed).hexdigest()
    encoded = base64.b85encode(compressed).decode("ascii")
    return encoded, digest


def _artifact_template(archive_b85: str, archive_digest: str) -> str:
    """Return the formatted artifact script."""

    wrapped_archive = _wrap_lines(archive_b85)
    template = f"""#!/usr/bin/env python3
\"\"\"Self-reconstructing artifact for the paper_quine repository.

The emitted script contains everything required to rebuild the original source
tree, including recomputation of the deterministic weight bundle directly from
its own bytes. Invoke ``--help`` for usage details.
\"\"\"

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import tarfile
import textwrap
import zlib
from pathlib import Path

ARCHIVE_B85 = \"\"\"{wrapped_archive}\"\"\"
ARCHIVE_SHA256 = \"{archive_digest}\"  # pragma: allowlist secret

START_TOKEN = \"{START_TOKEN}\"
END_TOKEN = \"{END_TOKEN}\"
OUTPUT_SCRIPT = Path(\"src/paper_quine/paper.tex\")
WEIGHTS_PATH = Path(\"src/paper_quine/data/weights.b85\")
WEIGHTS_DIGEST_PATH = Path(\"src/paper_quine/data/weights.sha256\")
ARTIFACT_DIGEST_PATH = Path(\"src/paper_quine/data/paper.sha256\")


def _decode_archive_bytes() -> bytes:
    compressed = base64.b85decode(ARCHIVE_B85.replace(\"\\n\", \"\").encode(\"ascii\"))
    digest = hashlib.sha256(compressed).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise RuntimeError(
            f\"Archive digest mismatch: expected {{ARCHIVE_SHA256}}, got {{digest}}\"
        )
    return zlib.decompress(compressed)


def _safe_extract(tar_stream: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar_stream.getmembers():
        target_path = destination / member.name
        if not target_path.resolve().is_relative_to(destination):
            raise RuntimeError(f\"Unsafe path in archive: {{member.name}}\")
    tar_stream.extractall(destination)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _wrap_text(value: str, width: int = 80) -> str:
    return \"\\n\".join(textwrap.wrap(value, width))


def _embed_weights(script_bytes: bytes) -> str:
    embedded = zlib.compress(script_bytes, level=9)
    embedding_b85 = base64.b85encode(embedded).decode(\"ascii\")
    digest = hashlib.sha256(script_bytes).hexdigest()
    payload = {{
        \"embedding_b85\": embedding_b85,
        \"embedding_sha256\": digest,
        \"end_token\": END_TOKEN,
        \"paper_sha256\": digest,
        \"start_token\": START_TOKEN,
        \"version\": 2,
    }}
    json_bytes = json.dumps(payload, separators=(\",\", \":\")).encode(\"utf-8\")
    compressed_payload = zlib.compress(json_bytes, level=9)
    weights_b85 = base64.b85encode(compressed_payload).decode(\"ascii\")
    return weights_b85


def rebuild(destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    archive_bytes = _decode_archive_bytes()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode=\"r:\") as tar:
        _safe_extract(tar, destination)

    script_source = Path(__file__).resolve()
    script_bytes = script_source.read_bytes()
    output_script = (destination / OUTPUT_SCRIPT).resolve()
    _ensure_parent(output_script)
    output_script.write_bytes(script_bytes)

    artifact_digest = hashlib.sha256(script_bytes).hexdigest() + \"\\n\"
    _ensure_parent(destination / ARTIFACT_DIGEST_PATH)
    (destination / ARTIFACT_DIGEST_PATH).write_text(artifact_digest, encoding=\"utf-8\")

    weights_b85 = _embed_weights(script_bytes)
    wrapped_weights = _wrap_text(weights_b85)
    raw_weights = wrapped_weights + \"\\n\"
    (destination / WEIGHTS_PATH).write_text(raw_weights, encoding=\"utf-8\")
    weights_digest = hashlib.sha256(raw_weights.encode(\"utf-8\")).hexdigest() + \"\\n\"
    (destination / WEIGHTS_DIGEST_PATH).write_text(weights_digest, encoding=\"utf-8\")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=\"Rebuild the paper_quine repository from this artifact.\"
    )
    parser.add_argument(
        \"--output\",
        type=Path,
        default=Path.cwd(),
        help=(
            \"Target directory for reconstruction \"
            \"(default: current working directory).\"
        ),
    )
    args = parser.parse_args()
    rebuild(args.output)
    print(f\"Repository reconstructed at {{args.output.resolve()}}\")
    return 0


if __name__ == \"__main__\":
    raise SystemExit(main())
"""

    return template


def _write_file(path: Path, data: bytes) -> None:
    """Write bytes to path relative to the repository root."""

    absolute = REPO_ROOT / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(data)


def _rebuild_assets() -> None:
    tracked = _git_tracked_files()
    archive_sources = [p for p in tracked if p not in GENERATED_PATHS]
    archive_b85, archive_digest = _build_archive(archive_sources)

    artifact_source = _artifact_template(archive_b85, archive_digest)
    artifact_bytes = artifact_source.encode("utf-8")
    _write_file(ARTIFACT_PATH, artifact_bytes)

    artifact_digest = hashlib.sha256(artifact_bytes).hexdigest() + "\n"
    _write_file(DIGEST_PATH, artifact_digest.encode("utf-8"))

    embedded = zlib.compress(artifact_bytes, level=9)
    embedding_b85 = base64.b85encode(embedded).decode("ascii")
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest()
    payload = {
        "embedding_b85": embedding_b85,
        "embedding_sha256": artifact_sha,
        "end_token": END_TOKEN,
        "paper_sha256": artifact_sha,
        "start_token": START_TOKEN,
        "version": 2,
    }
    payload_bytes = json.dumps(payload, separators=JSON_SEPARATORS).encode("utf-8")
    compressed_payload = zlib.compress(payload_bytes, level=9)
    weights_b85 = base64.b85encode(compressed_payload).decode("ascii")
    wrapped_weights = _wrap_lines(weights_b85)
    raw_weights = wrapped_weights + "\n"
    _write_file(WEIGHTS_PATH, raw_weights.encode("utf-8"))
    weights_digest = hashlib.sha256(raw_weights.encode("utf-8")).hexdigest() + "\n"
    _write_file(WEIGHTS_DIGEST_PATH, weights_digest.encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic AI quine artifact and weights."
    )
    _ = parser.parse_args()
    _rebuild_assets()
    print("Regenerated artifact script and weight bundle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
