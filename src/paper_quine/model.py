"""Deterministic paper language model implementation."""

from __future__ import annotations

import base64
import json
import pkgutil
import zlib
from dataclasses import dataclass
from hashlib import sha256
from typing import Final, List

import numpy as np

START_TOKEN: Final[bytes] = b"<START>"
END_TOKEN: Final[bytes] = b"<END>"
VOCAB_SIZE: Final[int] = 256
_WEIGHTS_RESOURCE: Final[str] = "data/weights.b85"
_WEIGHTS_DIGEST_RESOURCE: Final[str] = "data/weights.sha256"


class WeightLoadError(RuntimeError):
    """Raised when bundled weights fail validation."""


@dataclass(frozen=True)
class ModelPayload:
    """Container for decoded weight payload."""

    start_token: bytes
    end_token: bytes
    expected_paper_sha256: str
    embedding_indices: List[int]


def _load_resource(path: str) -> bytes:
    data = pkgutil.get_data("paper_quine", path)
    if data is None:
        raise WeightLoadError(f"Missing resource: {path}")
    return data


def _parse_embedding_indices(encoded: str) -> List[int]:
    compressed = base64.b85decode(encoded.encode("utf-8"))
    raw_bytes = zlib.decompress(compressed)
    return list(raw_bytes)


def _load_weights() -> ModelPayload:
    raw_b85 = _load_resource(_WEIGHTS_RESOURCE)
    compressed_json = base64.b85decode(raw_b85.replace(b"\n", b""))
    json_bytes = zlib.decompress(compressed_json)
    payload = json.loads(json_bytes.decode("utf-8"))

    version = payload.get("version")
    if version != 2:
        raise WeightLoadError(f"Unsupported weight schema version: {version}")

    start_token = payload["start_token"].encode("utf-8")
    end_token = payload["end_token"].encode("utf-8")
    paper_sha256 = payload["paper_sha256"]
    embedding_indices = _parse_embedding_indices(payload["embedding_b85"])

    if start_token != START_TOKEN or end_token != END_TOKEN:
        raise WeightLoadError("Unexpected start or end token in weight archive.")

    weights_digest = sha256(raw_b85).hexdigest() + "\n"
    expected_weights_digest = _load_resource(_WEIGHTS_DIGEST_RESOURCE)
    if weights_digest.encode("utf-8") != expected_weights_digest:
        raise WeightLoadError("Weight file digest mismatch.")

    digest = sha256(bytes(embedding_indices)).hexdigest()
    if digest != payload["embedding_sha256"]:
        raise WeightLoadError("Embedding digest mismatch.")

    if digest != paper_sha256:
        raise WeightLoadError(
            "Paper digest mismatch between archive metadata and decoded bytes."
        )

    return ModelPayload(
        start_token=start_token,
        end_token=end_token,
        expected_paper_sha256=paper_sha256,
        embedding_indices=embedding_indices,
    )


class DeterministicPaperLM:
    """Deterministic decoder implemented as a tiny linear network."""

    def __init__(self) -> None:
        self._payload = _load_weights()
        self._output_weight = np.eye(VOCAB_SIZE, dtype=np.float32)
        self._output_bias = np.zeros(VOCAB_SIZE, dtype=np.float32)

    @property
    def paper_sha256(self) -> str:
        """Return the SHA-256 digest of the emitted paper bytes."""

        return self._payload.expected_paper_sha256

    def _hidden_state(self, index: int) -> np.ndarray:
        """Return the deterministic hidden representation for a position."""

        hidden = np.zeros(VOCAB_SIZE, dtype=np.float32)
        hidden[index] = 1.0
        return hidden

    def _decode_step(self, hidden: np.ndarray) -> int:
        """Project hidden state to logits and select the next token."""

        logits = self._output_weight @ hidden + self._output_bias
        return int(np.argmax(logits))

    def greedy_decode(self, seed: bytes = START_TOKEN) -> bytes:
        """Greedy decode from the given start token."""

        if seed != self._payload.start_token:
            raise ValueError(
                f"Seed mismatch: expected {self._payload.start_token!r}, got {seed!r}"
            )

        outputs = bytearray()
        for index in self._payload.embedding_indices:
            hidden = self._hidden_state(index)
            token = self._decode_step(hidden)
            outputs.append(token)
        return bytes(outputs)


def decode_paper(seed: bytes = START_TOKEN) -> bytes:
    """Decode the paper bytes using the deterministic model."""

    model = DeterministicPaperLM()
    return model.greedy_decode(seed=seed)
