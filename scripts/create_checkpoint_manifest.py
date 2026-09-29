#!/usr/bin/env python3
"""Create the integrity manifest required for real-robot inference."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

MANIFEST_NAMES = {"checkpoint_manifest.sha256", "SHA256SUMS.expected"}


def digest_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(checkpoint: Path, output_name: str) -> str:
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint}")
    if output_name not in MANIFEST_NAMES:
        raise ValueError(f"output name must be one of {sorted(MANIFEST_NAMES)}")

    required = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
    missing = sorted(name for name in required if not (checkpoint / name).is_file())
    if missing:
        raise ValueError("checkpoint is incomplete: " + ", ".join(missing))

    files: list[Path] = []
    for path in checkpoint.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"checkpoint contains a symlink: {path}")
        if path.is_file() and path.name not in MANIFEST_NAMES:
            files.append(path)

    return "".join(
        f"{digest_file(path)}  {path.relative_to(checkpoint).as_posix()}\n"
        for path in sorted(files)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a sha256sum-compatible checkpoint integrity manifest"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--output-name",
        choices=sorted(MANIFEST_NAMES),
        default="SHA256SUMS.expected",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    content = build_manifest(checkpoint, args.output_name)
    output = checkpoint / args.output_name
    temporary = checkpoint / f".{args.output_name}.tmp"
    temporary.write_text(content)
    temporary.replace(output)
    print(f"Wrote {output} ({len(content.splitlines())} artifacts)")


if __name__ == "__main__":
    main()
