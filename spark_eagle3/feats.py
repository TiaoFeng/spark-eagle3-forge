"""Feature storage for EAGLE-3 extraction.

Features live in an anonymous memfd, mapped into this process. That keeps them
purely in RAM (no NVMe writes, no tmpfs quota), and because the shim writes
straight into the mapping there is no copy through Python.

Row layout, 15,368 bytes at the v8 geometry:

    0      bf16[n_layer_feat][n_embd]   input hidden states of the target layers
    -8     int32                       token[p+1]   (shift-left, see notes)
    -4     int32                       argmax[p+1]  (training label)
"""

from __future__ import annotations

import ctypes
import mmap
import os

import numpy as np

__all__ = ["row_bytes", "capacity_tokens", "FeatureStore"]


def row_bytes(n_layer_feat: int, n_embd: int) -> int:
    return n_layer_feat * n_embd * 2 + 8


def capacity_tokens(capacity_bytes: int, n_layer_feat: int, n_embd: int) -> int:
    return capacity_bytes // row_bytes(n_layer_feat, n_embd)


class FeatureStore:
    """memfd-backed array of fixed-size feature rows."""

    def __init__(self, capacity_bytes: int, n_layer_feat: int, n_embd: int,
                 name: str = "spark_feat"):
        self.row_bytes = row_bytes(n_layer_feat, n_embd)
        self.n_layer_feat = n_layer_feat
        self.n_embd = n_embd
        self.capacity_bytes = int(capacity_bytes)
        self.capacity = self.capacity_bytes // self.row_bytes

        self.fd = os.memfd_create(name, flags=0)
        os.ftruncate(self.fd, self.capacity_bytes)
        self.buf = mmap.mmap(self.fd, self.capacity_bytes)
        self.n_tokens = 0

    # -- capacity ----------------------------------------------------------
    def require(self, n_tokens: int) -> None:
        if n_tokens > self.capacity:
            need = n_tokens * self.row_bytes
            raise MemoryError(
                "feature store too small: %d tokens need %.1f GiB, have %.1f GiB"
                % (n_tokens, need / 2**30, self.capacity_bytes / 2**30))

    # -- writing -----------------------------------------------------------
    def row_ptr(self, token_index: int) -> int:
        if token_index >= self.capacity:
            raise IndexError("row %d beyond capacity %d" % (token_index, self.capacity))
        view = ctypes.c_char.from_buffer(self.buf, token_index * self.row_bytes)
        return ctypes.addressof(view)

    def _stride_view(self, token_index: int, offset_in_row: int) -> np.ndarray:
        return np.ndarray(shape=(1,), dtype="<i4", buffer=self.buf,
                          offset=token_index * self.row_bytes + offset_in_row,
                          strides=(self.row_bytes,))

    def write_tails(self, first_row: int, tokens_next: np.ndarray,
                    argmax_next: np.ndarray) -> None:
        """Fill token[p+1] / argmax[p+1] for a run of rows.

        A strided numpy view keeps this O(1) python-level work regardless of how
        many rows the chunk produced.
        """
        n = len(tokens_next)
        if n == 0:
            return
        tail = self.n_layer_feat * self.n_embd * 2
        tok_view = np.ndarray(shape=(n,), dtype="<i4", buffer=self.buf,
                              offset=first_row * self.row_bytes + tail,
                              strides=(self.row_bytes,))
        am_view = np.ndarray(shape=(n,), dtype="<i4", buffer=self.buf,
                             offset=first_row * self.row_bytes + tail + 4,
                             strides=(self.row_bytes,))
        tok_view[:] = tokens_next
        am_view[:] = argmax_next

    def row(self, i: int) -> np.ndarray:
        """Copy of one row as uint8."""
        start = i * self.row_bytes
        return np.frombuffer(self.buf, dtype=np.uint8, count=self.row_bytes, offset=start).copy()

    # -- lifecycle ---------------------------------------------------------
    def flush(self) -> None:
        self.buf.flush()

    def close(self) -> None:
        """Release the mapping even if a numpy view still holds an export."""
        try:
            self.buf.close()
        except BufferError:
            # a live numpy view pins the buffer; the mapping goes away with the
            # process anyway, and failing here would mask the real work
            pass
        finally:
            try:
                os.close(self.fd)
            except OSError:
                pass

    def __enter__(self) -> "FeatureStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
