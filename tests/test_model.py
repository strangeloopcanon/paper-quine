from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from paper_quine import model
from paper_quine.model import DeterministicPaperLM, decode_paper
from paper_quine.verify import main as verify_main
from paper_quine.verify import run_llm_live

REFERENCE_LATEX = Path("src/paper_quine/paper.tex")
EXPECTED_PAPER_DIGEST = (
    Path("src/paper_quine/data/paper.sha256").read_text(encoding="utf-8").strip()
)


def test_decode_matches_reference() -> None:
    decoded = decode_paper()
    expected = REFERENCE_LATEX.read_bytes()
    assert decoded == expected
    assert hashlib.sha256(decoded).hexdigest() == EXPECTED_PAPER_DIGEST


def test_model_rejects_wrong_seed() -> None:
    model = DeterministicPaperLM()
    with pytest.raises(ValueError):
        model.greedy_decode(seed=b"WRONG")


def test_llm_live_runner(tmp_path: Path) -> None:
    output_path = tmp_path / "paper.tex"
    exit_code = run_llm_live(output_path)
    assert exit_code == 0
    assert output_path.read_bytes() == REFERENCE_LATEX.read_bytes()


def test_model_reports_digest() -> None:
    model_instance = DeterministicPaperLM()
    assert model_instance.paper_sha256 == EXPECTED_PAPER_DIGEST
    assert np.allclose(  # noqa: SLF001
        model_instance._output_weight,
        np.eye(model.VOCAB_SIZE, dtype=np.float32),  # noqa: SLF001
    )


def test_cli_main_llm_live(tmp_path: Path) -> None:
    output_path = tmp_path / "paper.tex"
    exit_code = verify_main(["--llm-live", "--output", str(output_path)])
    assert exit_code == 0


def test_cli_main_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = verify_main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Paper quine verification tooling" in captured.out


def test_load_weights_detects_missing_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    original_get_data = model.pkgutil.get_data

    def fake_get_data(package: str, resource: str) -> bytes | None:
        if resource == model._WEIGHTS_RESOURCE:  # noqa: SLF001
            return None
        return original_get_data(package, resource)

    monkeypatch.setattr(model.pkgutil, "get_data", fake_get_data)

    with pytest.raises(model.WeightLoadError):
        model._load_weights()  # noqa: SLF001


def test_load_weights_detects_digest_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    original_loader = model._load_resource  # type: ignore[attr-defined]

    def fake_loader(path: str) -> bytes:
        data = original_loader(path)
        if path == model._WEIGHTS_DIGEST_RESOURCE:  # noqa: SLF001
            return b"deadbeef\n"
        return data

    monkeypatch.setattr(model, "_load_resource", fake_loader)

    with pytest.raises(model.WeightLoadError):
        model._load_weights()  # noqa: SLF001


def test_hidden_state_is_one_hot() -> None:
    lm = DeterministicPaperLM()
    vector = lm._hidden_state(42)  # noqa: SLF001
    assert vector.shape == (model.VOCAB_SIZE,)
    assert np.count_nonzero(vector) == 1
    assert vector[42] == pytest.approx(1.0)
