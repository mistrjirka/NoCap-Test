#!/usr/bin/env python3
"""Download the official pre-tokenized FineWeb shards used by NoCap."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "kjj0/fineweb10B-gpt2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fineweb10B"),
        help="Directory containing the .bin files",
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        default=50,
        help="1 for a quick test; 50 for the complete official 5B-token budget",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Download only the fixed public validation shard",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face dataset revision; use a commit hash for immutable runs",
    )
    args = parser.parse_args()
    if not 0 <= args.train_shards <= 50:
        raise ValueError("--train-shards must be between 0 and 50")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    filenames = ["fineweb_val_000000.bin"]
    if not args.validation_only:
        filenames.extend(
            f"fineweb_train_{index:06d}.bin"
            for index in range(1, args.train_shards + 1)
        )

    print(
        f"Downloading {len(filenames)} file(s) from dataset {REPO_ID} to "
        f"{args.output_dir.resolve()}"
    )
    if args.train_shards == 50:
        print("The training shards contain 5B uint16 tokens: expect roughly 10 GB.")

    for position, filename in enumerate(filenames, start=1):
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            revision=args.revision,
            local_dir=str(args.output_dir),
        )
        print(f"[{position}/{len(filenames)}] {path}")


if __name__ == "__main__":
    main()
