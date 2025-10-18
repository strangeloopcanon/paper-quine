"""Deterministic paper language model implementation."""

from __future__ import annotations

import base64
import json
import pkgutil
import zlib
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

START_TOKEN: Final[bytes] = b"<START>"
END_TOKEN: Final[bytes] = b"<END>"
_WEIGHTS_RESOURCE: Final[str] = "data/weights.b85"
_WEIGHTS_DIGEST_RESOURCE: Final[str] = "data/weights.sha256"


class WeightLoadError(RuntimeError):
    """Raised when bundled weights fail validation."""


@dataclass(frozen=True)
class ModelPayload:
    """Container for decoded weight payload."""

    start_token: bytes
    end_token: bytes
    paper_bytes: bytes
    expected_paper_sha256: str


def _load_resource(path: str) -> bytes:
    data = pkgutil.get_data("paper_quine", path)
    if data is None:
        raise WeightLoadError(f"Missing resource: {path}")
    return data


def _load_weights() -> ModelPayload:
    raw_b85 = _load_resource(_WEIGHTS_RESOURCE)
    compressed_json = base64.b85decode(raw_b85.replace(b"\n", b""))
    json_bytes = zlib.decompress(compressed_json)
    payload = json.loads(json_bytes.decode("utf-8"))

    start_token = payload["start_token"].encode("utf-8")
    end_token = payload["end_token"].encode("utf-8")
    paper_sha256 = payload["paper_sha256"]
    compressed_paper = base64.b85decode(payload["paper_z85"].encode("utf-8"))
    paper_bytes = zlib.decompress(compressed_paper)

    if start_token != START_TOKEN or end_token != END_TOKEN:
        raise WeightLoadError("Unexpected start or end token in weight archive.")

    digest = sha256(paper_bytes).hexdigest()
    if digest != paper_sha256:
        raise WeightLoadError(
            "Paper digest mismatch between archive metadata and decoded bytes."
        )

    weights_digest = sha256(raw_b85).hexdigest() + "\n"
    expected_weights_digest = _load_resource(_WEIGHTS_DIGEST_RESOURCE)
    if weights_digest.encode("utf-8") != expected_weights_digest:
        raise WeightLoadError("Weight file digest mismatch.")

    return ModelPayload(
        start_token=start_token,
        end_token=end_token,
        paper_bytes=paper_bytes,
        expected_paper_sha256=paper_sha256,
    )


class DeterministicPaperLM:
    """Deterministic decoder over the embedded paper bytes."""

    def __init__(self) -> None:
        self._payload = _load_weights()

    @property
    def paper_sha256(self) -> str:
        """Return the SHA-256 digest of the emitted paper bytes."""

        return self._payload.expected_paper_sha256

    def greedy_decode(self, seed: bytes = START_TOKEN) -> bytes:
        """Greedy decode from the given start token."""

        if seed != self._payload.start_token:
            raise ValueError(
                f"Seed mismatch: expected {self._payload.start_token!r}, got {seed!r}"
            )
        return self._payload.paper_bytes


def decode_paper(seed: bytes = START_TOKEN) -> bytes:
    """Decode the paper bytes using the deterministic model."""

    model = DeterministicPaperLM()
    return model.greedy_decode(seed=seed)
