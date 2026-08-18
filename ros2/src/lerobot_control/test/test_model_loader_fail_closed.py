"""Fail-closed checkpoint loading tests."""

import json
import sys
from hashlib import sha256
from types import ModuleType, SimpleNamespace

import pytest
import torch
from lerobot_control.model_loader import ModelLoader
from safetensors.torch import save_file


def write_pi05_checkpoint(path, *, include_processors=True, include_weights=True) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"type": "pi05"}))
    if include_processors:
        (path / "policy_preprocessor.json").write_text(json.dumps({"steps": ["pre"]}))
        (path / "policy_postprocessor.json").write_text(json.dumps({"steps": ["post"]}))
    if include_weights:
        save_file({"weight": torch.tensor([1.0])}, path / "model.safetensors")


def write_manifest(path, *, corrupt_weight=False) -> None:
    lines = []
    for artifact in sorted(item for item in path.iterdir() if item.is_file()):
        digest = sha256(artifact.read_bytes()).hexdigest()
        if corrupt_weight and artifact.name == "model.safetensors":
            digest = "0" * 64
        lines.append(f"{digest}  {artifact.name}\n")
    (path / "SHA256SUMS.expected").write_text("".join(lines))


def test_pi05_requires_weights_and_saved_processors(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    write_pi05_checkpoint(checkpoint, include_weights=False)

    with pytest.raises(RuntimeError, match="model.safetensors"):
        ModelLoader(str(checkpoint), model_type="pi05", require_checkpoint_manifest=False)

    (checkpoint / "model.safetensors").write_bytes(b"not safetensors")
    with pytest.raises(RuntimeError, match="weights are unreadable"):
        ModelLoader(str(checkpoint), model_type="pi05", require_checkpoint_manifest=False)


def test_pi05_requires_both_processor_configs(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    write_pi05_checkpoint(checkpoint, include_processors=False)

    with pytest.raises(RuntimeError, match="policy_preprocessor.json"):
        ModelLoader(str(checkpoint), model_type="pi05", require_checkpoint_manifest=False)


def test_manifest_covers_required_files_and_verifies_checksums(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    write_pi05_checkpoint(checkpoint)
    write_manifest(checkpoint, corrupt_weight=True)

    with pytest.raises(RuntimeError, match="checksum mismatch for model.safetensors"):
        ModelLoader(str(checkpoint), model_type="pi05")

    (checkpoint / "SHA256SUMS.expected").unlink()
    write_manifest(checkpoint)
    loader = ModelLoader(str(checkpoint), model_type="pi05")
    assert loader.model_path == checkpoint


def test_manifest_cannot_escape_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    write_pi05_checkpoint(checkpoint)
    (checkpoint / "SHA256SUMS.expected").write_text(f"{'0' * 64}  ../outside\n")

    with pytest.raises(RuntimeError, match="Unsafe or invalid"):
        ModelLoader(str(checkpoint), model_type="pi05")


def install_fake_pi05_modules(monkeypatch, *, complete_weight_load: bool) -> None:
    class FakePI05Policy:
        def __init__(self):
            self.config = SimpleNamespace(n_action_steps=50)

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            model = cls()
            if complete_weight_load:
                model.load_state_dict({}, strict=True)
            return model

        def load_state_dict(self, _state_dict, *_args, **_kwargs):
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        def to(self, _device):
            return self

        def eval(self):
            return self

    class FakeConfig:
        compile_model = True

        @classmethod
        def from_pretrained(cls, _path):
            return cls()

    pi05_module = ModuleType("lerobot.policies.pi05")
    pi05_module.PI05Policy = FakePI05Policy
    configs_module = ModuleType("lerobot.configs.policies")
    configs_module.PreTrainedConfig = FakeConfig
    monkeypatch.setitem(sys.modules, "lerobot.policies.pi05", pi05_module)
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", configs_module)


def test_pi05_rejects_the_random_model_returned_after_swallowed_load_error(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    write_pi05_checkpoint(checkpoint)
    install_fake_pi05_modules(monkeypatch, complete_weight_load=False)
    loader = ModelLoader(
        str(checkpoint),
        model_type="pi05",
        require_checkpoint_manifest=False,
    )

    with pytest.raises(RuntimeError, match="random/partial weights"):
        loader.load()


def test_pi05_accepts_only_a_completed_exact_state_dict_load(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint"
    write_pi05_checkpoint(checkpoint)
    install_fake_pi05_modules(monkeypatch, complete_weight_load=True)
    loader = ModelLoader(
        str(checkpoint),
        model_type="pi05",
        require_checkpoint_manifest=False,
    )

    model = loader.load()

    assert model._anvil_checkpoint_load_completed is True


def test_pi05_processor_load_error_has_no_unnormalized_fallback(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint"
    write_pi05_checkpoint(checkpoint)
    loader = ModelLoader(
        str(checkpoint),
        model_type="pi05",
        require_checkpoint_manifest=False,
    )
    monkeypatch.setattr(
        loader,
        "load",
        lambda: SimpleNamespace(config=SimpleNamespace()),
    )

    class FailingPipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise RuntimeError("processor data corrupt")

    processor_module = ModuleType("lerobot.processor")
    processor_module.PolicyProcessorPipeline = FailingPipeline
    pi05_processor_module = ModuleType("lerobot.policies.pi05.processor_pi05")
    monkeypatch.setitem(sys.modules, "lerobot.processor", processor_module)
    monkeypatch.setitem(
        sys.modules,
        "lerobot.policies.pi05.processor_pi05",
        pi05_processor_module,
    )

    with pytest.raises(RuntimeError, match="refusing inference"):
        loader.load_with_processors()
