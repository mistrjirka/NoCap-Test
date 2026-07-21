"""Reader for the pre-tokenized FineWeb binary shards used by NoCap."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

MAGIC = 20240520
HEADER_INTS = 256


def _peek_data_shard(filename: str) -> int:
    with open(filename, "rb") as handle:
        header = np.frombuffer(handle.read(HEADER_INTS * 4), dtype=np.int32)
    if len(header) != HEADER_INTS or header[0] != MAGIC:
        raise ValueError(f"Invalid FineWeb shard header: {filename}")
    if header[1] != 1:
        raise ValueError(f"Unsupported FineWeb shard version {header[1]}: {filename}")
    return int(header[2])


def _load_data_shard(filename: str) -> np.ndarray:
    with open(filename, "rb") as handle:
        header = np.frombuffer(handle.read(HEADER_INTS * 4), dtype=np.int32)
        if len(header) != HEADER_INTS or header[0] != MAGIC or header[1] != 1:
            raise ValueError(f"Invalid FineWeb shard: {filename}")
        expected_tokens = int(header[2])
        tokens = np.frombuffer(handle.read(), dtype=np.uint16)
    if len(tokens) != expected_tokens:
        raise ValueError(
            f"Shard {filename} claims {expected_tokens:,} tokens but contains "
            f"{len(tokens):,}"
        )
    return tokens


class DistributedDataLoader:
    def __init__(
        self,
        filename_pattern: str,
        batch_size: int,
        sequence_length: int,
        process_rank: int,
        num_processes: int,
        device: torch.device,
    ) -> None:
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device
        self.files = sorted(glob.glob(filename_pattern))
        if not self.files:
            raise FileNotFoundError(
                f"No shards match {filename_pattern!r}. Run download_data.py first."
            )

        total = 0
        for filename in self.files:
            tokens = _peek_data_shard(filename)
            if tokens < num_processes * batch_size * sequence_length + 1:
                raise ValueError(f"Shard too small for requested batch shape: {filename}")
            total += tokens
        self.total_tokens = total
        self.current_shard = 0
        self.current_position = 0
        self.tokens: np.ndarray
        self.reset()

    def describe(self) -> str:
        return (
            f"{self.total_tokens:,} tokens across {len(self.files)} shard(s); "
            f"B={self.batch_size}, T={self.sequence_length}"
        )

    def set_batch_shape(self, batch_size: int, sequence_length: int) -> None:
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("batch_size and sequence_length must be positive")
        old_rank_offset = (
            self.process_rank * self.batch_size * self.sequence_length
        )
        global_position = max(0, self.current_position - old_rank_offset)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        new_rank_offset = self.process_rank * batch_size * sequence_length
        self.current_position = global_position + new_rank_offset
        needed = self.num_processes * batch_size * sequence_length + 1
        if self.current_position + needed > len(self.tokens):
            self.advance()

    def reset(self) -> None:
        self.current_shard = 0
        self.current_position = (
            self.process_rank * self.batch_size * self.sequence_length
        )
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self) -> None:
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = (
            self.process_rank * self.batch_size * self.sequence_length
        )
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = self.batch_size, self.sequence_length
        count = batch_size * sequence_length
        end = self.current_position + count + 1
        if end > len(self.tokens):
            self.advance()
            end = self.current_position + count + 1

        buffer = self.tokens[self.current_position:end]
        # uint16 -> int64 is required for embedding indices.
        tensor = torch.from_numpy(buffer.astype(np.int64, copy=True))
        x = tensor[:-1].view(batch_size, sequence_length)
        y = tensor[1:].view(batch_size, sequence_length)

        self.current_position += count * self.num_processes
        if self.current_position + count * self.num_processes + 1 > len(self.tokens):
            self.advance()

        return (
            x.to(self.device, non_blocking=True),
            y.to(self.device, non_blocking=True),
        )


def assert_disjoint_patterns(train_pattern: str, val_pattern: str) -> None:
    train_files = {Path(path).resolve() for path in glob.glob(train_pattern)}
    val_files = {Path(path).resolve() for path in glob.glob(val_pattern)}
    overlap = train_files & val_files
    if overlap:
        joined = ", ".join(str(path) for path in sorted(overlap))
        raise ValueError(f"Training and validation shards overlap: {joined}")
