"""Model loading utilities for LeRobot models.

Simplified model loader that uses LeRobot's built-in PolicyProcessorPipeline
for pre/post processing instead of custom implementations.
"""

import json
import random
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_deterministic_mode(seed: int = 42):
    """
    Set PyTorch to deterministic mode for reproducible inference.

    This should be called before loading the model for consistent results.

    Args:
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch, "use_deterministic_algorithms"):
        with suppress(Exception):
            torch.use_deterministic_algorithms(True, warn_only=True)


def reset_model_state(model):
    """
    Reset model's internal state for reproducible inference.

    For ACT models, this clears the action queue so each inference
    starts fresh with a new action chunk prediction.

    Args:
        model: The policy model (ACTPolicy, DiffusionPolicy, etc.)
    """
    if hasattr(model, "reset"):
        model.reset()


class ModelLoader:
    """
    Load trained LeRobot models from checkpoints.

    Uses LeRobot's built-in PolicyProcessorPipeline for preprocessing
    and postprocessing, eliminating the need for custom implementations.

    Supports:
    - ACT (Action Chunking Transformer)
    - Diffusion Policy
    - SmolVLA (Small Vision-Language-Action)
    - Pi0 (Physical Intelligence)
    - Pi0.5 (Physical Intelligence)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: str | None = None,
        logger=None,
        deterministic: bool = False,
        seed: int = 42,
        config_overrides: dict = None,
        rtc_config_yaml: dict = None,
        require_checkpoint_manifest: bool = True,
    ):
        """
        Initialize model loader.

        Args:
            model_path: Path to model checkpoint directory
            device: Device for inference ("cuda" or "cpu")
            model_type: Model type ("act", "diffusion", "smolvla", "pi0", "pi05").
                        None = auto-detect from config.json.
            logger: Optional ROS2 logger
            deterministic: If True, enable deterministic mode
            seed: Random seed for deterministic mode
            config_overrides: Dict of config values to override at load time
                e.g. {"temporal_ensemble_coeff": 0.01, "n_action_steps": 1}
            rtc_config_yaml: Dict from the ``rtc:`` YAML section. When set and
                the model is a VLA (pi0/pi05/smolvla), RTCConfig is injected
                after loading and ``model.init_rtc_processor()`` is called.
            require_checkpoint_manifest: Require and verify
                ``checkpoint_manifest.sha256`` before loading local weights.
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model_type = model_type
        self.logger = logger
        self.deterministic = deterministic
        self.seed = seed
        self.config_overrides = config_overrides or {}
        self.rtc_config_yaml = rtc_config_yaml or {}
        self.require_checkpoint_manifest = require_checkpoint_manifest
        self._model = None
        self._pre_processor = None
        self._post_processor = None
        self._orig_n_action_steps: int | None = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Auto-detect pretrained_model subdirectory.
        # Accepts paths at checkpoint step level (e.g. checkpoints/last or checkpoints/100000)
        # by checking for a pretrained_model subfolder when config.json is not in the given path.
        if not (self.model_path / "config.json").exists():
            pretrained_model_path = self.model_path / "pretrained_model"
            if pretrained_model_path.exists() and (pretrained_model_path / "config.json").exists():
                self.model_path = pretrained_model_path

        # Auto-detect model type from checkpoint if not provided
        if self.model_type is None:
            self.model_type = self._detect_model_type()

        self._validate_checkpoint_artifacts()

    def _log(self, level: str, msg: str):
        """Log message using ROS2 logger or print."""
        if self.logger:
            try:
                getattr(self.logger, level)(msg)
            except ValueError:
                # ROS2 logger can fail with "severity cannot be changed between calls"
                print(f"[{level.upper()}] {msg}")
        else:
            print(f"[{level.upper()}] {msg}")

    def _detect_model_type(self) -> str | None:
        """Read model type from config.json in the checkpoint directory."""
        config_path = self.model_path / "config.json"
        if config_path.exists():
            return json.loads(config_path.read_text()).get("type")
        return None

    def _validate_checkpoint_artifacts(self) -> None:
        """Fail before model construction if local checkpoint artifacts are unsafe."""
        if self.model_type not in {"pi0", "pi05", "smolvla"}:
            return

        required = ["config.json", "model.safetensors"]
        if self.model_type == "pi05":
            required.extend(["policy_preprocessor.json", "policy_postprocessor.json"])

        missing = [name for name in required if not (self.model_path / name).is_file()]
        if missing:
            raise RuntimeError(
                "Checkpoint is incomplete; required artifact(s) missing: "
                + ", ".join(missing)
            )

        for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
            path = self.model_path / name
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Checkpoint JSON is unreadable: {path}: {exc}") from exc
            if not isinstance(payload, dict) or not payload:
                raise RuntimeError(f"Checkpoint JSON must contain a non-empty object: {path}")

        config_type = json.loads((self.model_path / "config.json").read_text()).get("type")
        if config_type != self.model_type:
            raise RuntimeError(
                f"Checkpoint type mismatch: requested {self.model_type!r}, "
                f"config.json contains {config_type!r}"
            )

        weight_path = self.model_path / "model.safetensors"
        if weight_path.stat().st_size <= 8:
            raise RuntimeError(f"Checkpoint weight file is empty or truncated: {weight_path}")
        try:
            from safetensors import safe_open

            with safe_open(weight_path, framework="pt", device="cpu") as handle:
                tensor_names = list(handle.keys())
        except Exception as exc:
            raise RuntimeError(f"Checkpoint weights are unreadable: {weight_path}: {exc}") from exc
        if not tensor_names:
            raise RuntimeError(f"Checkpoint has no tensors: {weight_path}")

        manifest = next(
            (
                candidate
                for candidate in (
                    self.model_path / "checkpoint_manifest.sha256",
                    self.model_path / "SHA256SUMS.expected",
                )
                if candidate.is_file()
            ),
            None,
        )
        if self.require_checkpoint_manifest and manifest is None:
            raise RuntimeError(
                f"Checkpoint integrity manifest missing in {self.model_path}; expected "
                "checkpoint_manifest.sha256 or SHA256SUMS.expected. "
                "Generate it only after the isolated checkpoint copy is complete."
            )
        if manifest is not None:
            self._verify_checkpoint_manifest(manifest, required)

    def _verify_checkpoint_manifest(self, manifest: Path, required: list[str]) -> None:
        """Verify a sha256sum-compatible manifest without allowing path escape."""
        try:
            lines = manifest.read_text().splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"Checkpoint manifest is unreadable: {manifest}: {exc}") from exc

        entries: dict[str, str] = {}
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise RuntimeError(
                    f"Invalid checkpoint manifest line {line_number}: {line!r}"
                )
            digest, relative_name = parts
            relative_name = relative_name.removeprefix("*")
            relative = Path(relative_name)
            if (
                len(digest) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in digest)
                or relative.is_absolute()
                or ".." in relative.parts
                or relative_name in {"checkpoint_manifest.sha256", "SHA256SUMS.expected"}
            ):
                raise RuntimeError(
                    f"Unsafe or invalid checkpoint manifest line {line_number}: {line!r}"
                )
            normalized_name = relative.as_posix()
            if normalized_name in entries:
                raise RuntimeError(
                    f"Duplicate checkpoint manifest entry: {normalized_name}"
                )
            entries[normalized_name] = digest.lower()

        missing_entries = [name for name in required if name not in entries]
        if missing_entries:
            raise RuntimeError(
                "Checkpoint manifest does not cover required artifact(s): "
                + ", ".join(missing_entries)
            )

        for relative_name, expected in sorted(entries.items()):
            path = self.model_path / relative_name
            if path.is_symlink():
                raise RuntimeError(
                    f"Checkpoint manifest artifact must not be a symlink: {relative_name}"
                )
            if not path.is_file():
                raise RuntimeError(f"Checkpoint manifest artifact missing: {relative_name}")
            digest = sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            actual = digest.hexdigest()
            if actual != expected:
                raise RuntimeError(
                    f"Checkpoint checksum mismatch for {relative_name}: "
                    f"expected {expected}, got {actual}"
                )
        self._log("info", f"Verified checkpoint manifest ({len(entries)} artifacts)")

    @property
    def checkpoint_n_action_steps(self) -> int | None:
        """Original n_action_steps from checkpoint before any overrides were applied."""
        return self._orig_n_action_steps

    @property
    def chunk_size(self) -> int | None:
        """Get model chunk size from config."""
        if (
            self._model
            and hasattr(self._model, "config")
            and hasattr(self._model.config, "chunk_size")
        ):
            return self._model.config.chunk_size
        return None

    @property
    def n_action_steps(self) -> int | None:
        """Get model n_action_steps from config."""
        if (
            self._model
            and hasattr(self._model, "config")
            and hasattr(self._model.config, "n_action_steps")
        ):
            return self._model.config.n_action_steps
        return None

    def load(self):
        """
        Load model from checkpoint.

        Returns:
            Loaded model in eval mode
        """
        self._log("info", f"Loading {self.model_type} model from: {self.model_path}")

        if self.deterministic:
            set_deterministic_mode(self.seed)
            self._log("info", f"Deterministic mode enabled with seed={self.seed}")

        try:
            if self.model_type == "act":
                from lerobot.policies.act.modeling_act import ACTPolicy

                model = ACTPolicy.from_pretrained(str(self.model_path))
            elif self.model_type == "diffusion":
                from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

                model = DiffusionPolicy.from_pretrained(str(self.model_path))
            elif self.model_type == "smolvla":
                from lerobot.configs.policies import PreTrainedConfig
                from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

                vla_cfg = PreTrainedConfig.from_pretrained(str(self.model_path))
                if hasattr(vla_cfg, "compile_model"):
                    vla_cfg.compile_model = False
                model = SmolVLAPolicy.from_pretrained(str(self.model_path), config=vla_cfg)
            elif self.model_type == "pi0":
                from lerobot.configs.policies import PreTrainedConfig
                from lerobot.policies.pi0.modeling_pi0 import PI0Policy

                vla_cfg = PreTrainedConfig.from_pretrained(str(self.model_path))
                if hasattr(vla_cfg, "compile_model"):
                    vla_cfg.compile_model = False
                model = PI0Policy.from_pretrained(str(self.model_path), config=vla_cfg)
            elif self.model_type == "pi05":
                from lerobot.configs.policies import PreTrainedConfig
                from lerobot.policies.pi05 import PI05Policy

                class FailClosedPI05Policy(PI05Policy):
                    """Expose whether LeRobot actually completed state-dict loading."""

                    def load_state_dict(self, state_dict, *args, **kwargs):
                        result = super().load_state_dict(state_dict, *args, **kwargs)
                        if result.missing_keys or result.unexpected_keys:
                            raise RuntimeError(
                                "Pi0.5 checkpoint did not load exactly: "
                                f"missing={result.missing_keys}, "
                                f"unexpected={result.unexpected_keys}"
                            )
                        self._anvil_checkpoint_load_completed = True
                        return result

                vla_cfg = PreTrainedConfig.from_pretrained(str(self.model_path))
                if hasattr(vla_cfg, "compile_model"):
                    vla_cfg.compile_model = False
                model = FailClosedPI05Policy.from_pretrained(
                    str(self.model_path),
                    config=vla_cfg,
                    local_files_only=True,
                    strict=True,
                )
                # LeRobot 0.5.1 PI05Policy catches its own load exceptions and
                # returns a randomly initialized model. Never trust mere return.
                if not getattr(model, "_anvil_checkpoint_load_completed", False):
                    raise RuntimeError(
                        "Pi0.5 from_pretrained returned without completing strict "
                        "checkpoint weight loading; refusing random/partial weights"
                    )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            model = model.to(self.device)
            model.eval()
            self._model = model
            self._log("info", f"Model loaded successfully on {self.device}")

            # Snapshot checkpoint n_action_steps before any overrides
            if hasattr(model, "config"):
                self._orig_n_action_steps = getattr(model.config, "n_action_steps", None)

            # Apply config overrides after loading
            self._apply_config_overrides(model)

            # Inject RTCConfig for VLA models (pi0 / pi05 / smolvla)
            self._apply_rtc_config(model)

            return model

        except ImportError as e:
            self._log("error", f"lerobot package not installed: {e}")
            raise
        except Exception as e:
            self._log("error", f"Failed to load model: {e}")
            raise

    def _apply_config_overrides(self, model) -> None:
        """Apply config overrides after model is loaded.

        For temporal_ensemble_coeff, we also need to create the ensembler
        since it's only created in __init__ if the coeff was set at load time.
        """
        if not self.config_overrides or not hasattr(model, "config"):
            return

        config = model.config
        overrides_applied = []

        for key, value in self.config_overrides.items():
            if value is not None and hasattr(config, key):
                old_val = getattr(config, key)
                setattr(config, key, value)
                overrides_applied.append(f"{key}: {old_val} -> {value}")

        if overrides_applied:
            self._log("info", "Applied config overrides:")
            for override in overrides_applied:
                self._log("info", f"  - {override}")

        # Special handling: create temporal_ensembler if enabling at runtime
        if "temporal_ensemble_coeff" in self.config_overrides:
            coeff = self.config_overrides["temporal_ensemble_coeff"]
            if coeff is not None and not hasattr(model, "temporal_ensembler"):
                self._create_temporal_ensembler(model, coeff)

        # Special handling: propagate num_inference_steps to DiffusionModel internal cache
        # (DiffusionModel caches this at __init__ before overrides are applied)
        if "num_inference_steps" in self.config_overrides and hasattr(model, "diffusion"):
            model.diffusion.num_inference_steps = self.config_overrides["num_inference_steps"]

    def _apply_rtc_config(self, model) -> None:
        """Inject RTCConfig into VLA models and call init_rtc_processor().

        Only applies to pi0 / pi05 / smolvla. ACT and Diffusion are skipped
        silently. Must be called *after* _apply_config_overrides so that any
        n_action_steps override is already in place before RTC initialisation.
        """
        if self.model_type not in {"pi0", "pi05", "smolvla"}:
            return
        if not self.rtc_config_yaml:
            return

        try:
            from lerobot.configs.types import RTCAttentionSchedule
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            schedule_str = self.rtc_config_yaml.get("prefix_attention_schedule", "EXP")
            schedule = RTCAttentionSchedule[schedule_str]

            model.config.rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=self.rtc_config_yaml.get("execution_horizon", 10),
                max_guidance_weight=self.rtc_config_yaml.get("max_guidance_weight", 10.0),
                prefix_attention_schedule=schedule,
            )
            model.init_rtc_processor()
            self._log(
                "info",
                f"RTC enabled for {self.model_type} "
                f"(execution_horizon={model.config.rtc_config.execution_horizon}, "
                f"max_guidance_weight={model.config.rtc_config.max_guidance_weight}, "
                f"schedule={schedule_str})",
            )
        except Exception as e:
            self._log("error", f"Failed to initialise RTC: {e}")
            raise

    def _create_temporal_ensembler(self, model, coeff: float) -> None:
        """Create temporal ensembler for ACT model."""
        try:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

            chunk_size = model.config.chunk_size
            model.temporal_ensembler = ACTTemporalEnsembler(coeff, chunk_size)
            self._log(
                "info", f"Created temporal ensembler (coeff={coeff}, chunk_size={chunk_size})"
            )
        except ImportError as e:
            self._log("error", f"Failed to import ACTTemporalEnsembler: {e}")
        except Exception as e:
            self._log("error", f"Failed to create temporal ensembler: {e}")

    def load_with_processors(self) -> tuple[Any, Any, Any]:
        """
        Load model with LeRobot's built-in processor pipelines.

        Returns:
            Tuple of (model, preprocessor_pipeline, postprocessor_pipeline)
        """
        model = self.load()

        pre_processor = None
        post_processor = None

        try:
            from lerobot.processor import PolicyProcessorPipeline

            # VLA models register custom ProcessorStep implementations in their own
            # processor_<model>.py modules. These must be imported before calling
            # from_pretrained so that ProcessorStepRegistry recognises the step names
            # (e.g. "pi05_prepare_state_tokenizer_processor_step"). Without this import
            # the registry lookup fails and processor loading silently falls back to None.
            if self.model_type == "pi0":
                import lerobot.policies.pi0.processor_pi0  # noqa: F401
            elif self.model_type == "pi05":
                import lerobot.policies.pi05.processor_pi05  # noqa: F401
            elif self.model_type == "smolvla":
                import lerobot.policies.smolvla.processor_smolvla  # noqa: F401

            # Check if processor files exist
            pre_config = self.model_path / "policy_preprocessor.json"
            post_config = self.model_path / "policy_postprocessor.json"

            if pre_config.exists():
                pre_processor = PolicyProcessorPipeline.from_pretrained(
                    str(self.model_path), config_filename="policy_preprocessor.json"
                )
                # Move to device if the pipeline supports it
                if hasattr(pre_processor, "to") and callable(pre_processor.to):
                    with suppress(TypeError, AttributeError):
                        pre_processor.to(self.device)
                self._pre_processor = pre_processor
                self._log("info", "Loaded preprocessor pipeline from checkpoint")

            if post_config.exists():
                post_processor = PolicyProcessorPipeline.from_pretrained(
                    str(self.model_path), config_filename="policy_postprocessor.json"
                )
                # Move to device if the pipeline supports it
                if hasattr(post_processor, "to") and callable(post_processor.to):
                    with suppress(TypeError, AttributeError):
                        post_processor.to(self.device)
                self._post_processor = post_processor
                self._log("info", "Loaded postprocessor pipeline from checkpoint")

        except FileNotFoundError as e:
            if self.model_type == "pi05":
                raise RuntimeError(
                    "Pi0.5 processor pipeline artifact is missing; refusing "
                    "unnormalized inference"
                ) from e
            self._log("warn", f"Processor pipelines not found: {e}")
        except Exception as e:
            if self.model_type == "pi05":
                raise RuntimeError(
                    "Pi0.5 processor pipeline failed to load; refusing inference"
                ) from e
            self._log("error", f"Failed to load processor pipelines: {e}")

        if self.model_type == "pi05" and (
            pre_processor is None or post_processor is None
        ):
            missing = []
            if pre_processor is None:
                missing.append("preprocessor")
            if post_processor is None:
                missing.append("postprocessor")
            raise RuntimeError(
                "Pi0.5 requires checkpoint normalization processors; failed to load: "
                + ", ".join(missing)
            )

        if pre_processor is None and post_processor is None:
            self._log(
                "warn",
                "No processor pipelines found in checkpoint — observations and actions will NOT be "
                "normalized. If the model was trained with a non-IDENTITY normalization_mapping "
                "(e.g. MEAN_STD), inference results will be incorrect. "
                "Re-train or ensure policy_preprocessor.json / policy_postprocessor.json exist in "
                f"{self.model_path}",
            )

        return model, pre_processor, post_processor

    @staticmethod
    def validate_model_path(model_path: str) -> bool:
        """
        Validate model checkpoint path.

        Args:
            model_path: Path to check

        Returns:
            True if path appears to be valid model checkpoint
        """
        path = Path(model_path)
        if not path.exists():
            return False

        has_config = (path / "config.json").exists()
        has_model = (path / "pretrained_model").exists() or (path / "model.safetensors").exists()
        return has_config or has_model
