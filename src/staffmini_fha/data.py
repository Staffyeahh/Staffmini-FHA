from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    """
    Reads a contiguous token bin (int32) and yields fixed-size causal LM examples.
    """

    def __init__(self, bin_path: str | Path, sequence_length: int, token_dtype: Any = np.int64) -> None:
        self.bin_path = Path(bin_path)
        self.sequence_length = sequence_length
        self.token_dtype = token_dtype
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Token bin not found: {self.bin_path}")
        self.tokens = np.memmap(self.bin_path, mode="r", dtype=np.int32)
        if self.tokens.shape[0] <= sequence_length:
            raise ValueError(
                f"Token bin has {self.tokens.shape[0]} tokens, which is too small for sequence_length={sequence_length}."
            )
        self.num_sequences = self.tokens.shape[0] // sequence_length

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.sequence_length
        end = start + self.sequence_length
        x = np.array(self.tokens[start:end], dtype=self.token_dtype, copy=True)
        return {
            "input_ids": torch.from_numpy(x),
        }


class ChatSFTDataset(Dataset):
    """Packed SFT dataset with loss masking for assistant-only training."""

    def __init__(
        self, token_bin: str | Path, mask_bin: str | Path, sequence_length: int
    ) -> None:
        self.token_bin = Path(token_bin)
        self.mask_bin = Path(mask_bin)
        self.sequence_length = sequence_length
        if not self.token_bin.exists():
            raise FileNotFoundError(f"Token bin not found: {self.token_bin}")
        if not self.mask_bin.exists():
            raise FileNotFoundError(f"Mask bin not found: {self.mask_bin}")
        self.tokens = np.memmap(self.token_bin, mode="r", dtype=np.int32)
        self.mask = np.memmap(self.mask_bin, mode="r", dtype=np.int8)
        if self.tokens.shape[0] != self.mask.shape[0]:
            raise ValueError(
                f"Token/mask size mismatch: {self.tokens.shape[0]} vs {self.mask.shape[0]}"
            )
        self.num_sequences = self.tokens.shape[0] // sequence_length

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.sequence_length
        end = start + self.sequence_length
        x = np.array(self.tokens[start:end], dtype=np.int64, copy=True)
        y = np.array(self.tokens[start:end], dtype=np.int64, copy=True)
        m = np.array(self.mask[start:end], dtype=np.float32, copy=True)
        return {
            "input_ids": torch.from_numpy(x),
            "labels": torch.from_numpy(y),
            "loss_mask": torch.from_numpy(m),
        }
