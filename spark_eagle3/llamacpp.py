"""ctypes binding for csrc/spark_feat.cpp.

The shim exists because `llama_get/set_embeddings_layer_inp()` live in
`src/llama-ext.h`, which is a private header with C++ linkage. Everything the
shim exports is plain C, so ctypes needs no struct-layout guessing beyond the
`sp_params` POD below.

Only the parts needed for tokenization and coverage work are wrapped here;
feature extraction goes through `sp_decode()` and is added when the extractor
lands. Build the shim first:

    cmake -S csrc -B csrc/build -DLLAMACPP_ROOT=$LLAMACPP_ROOT
    cmake --build csrc/build
"""

from __future__ import annotations

import ctypes as C
import os
import pathlib

import numpy as np

__all__ = ["ShimParams", "Shim", "find_shim"]


class ShimParams(C.Structure):
    _fields_ = [
        ("model_path", C.c_char_p),
        ("n_gpu_layers", C.c_int32),
        ("n_ctx", C.c_uint32),
        ("n_batch", C.c_uint32),
        ("n_ubatch", C.c_uint32),
        ("verbose", C.c_int32),
    ]


def find_shim() -> str:
    """Locate libspark_feat.so next to this source tree."""
    env = os.environ.get("SPARK_SHIM")
    if env:
        return env
    here = pathlib.Path(__file__).resolve().parent.parent
    for cand in (here / "csrc" / "build" / "libspark_feat.so",
                 here / "csrc" / "build" / "libspark_feat.dylib"):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        "libspark_feat.so not found; build csrc/ first or set SPARK_SHIM")


def _bind(lib: C.CDLL) -> None:
    sig = {
        "sp_create": (C.c_void_p, [C.POINTER(ShimParams)]),
        "sp_destroy": (None, [C.c_void_p]),
        "sp_last_error": (C.c_char_p, [C.c_void_p]),
        "sp_ok": (C.c_int32, [C.c_void_p]),
        "sp_n_embd": (C.c_int32, [C.c_void_p]),
        "sp_n_vocab": (C.c_int32, [C.c_void_p]),
        "sp_n_layer": (C.c_int32, [C.c_void_p]),
        "sp_set_layers": (C.c_int32, [C.c_void_p, C.POINTER(C.c_int32), C.c_int32]),
        "sp_n_layer_feat": (C.c_int32, [C.c_void_p]),
        "sp_layers": (C.POINTER(C.c_int32), [C.c_void_p]),
        "sp_row_stride": (C.c_int32, [C.c_void_p]),
        "sp_tokenize": (C.c_int32, [C.c_void_p, C.c_char_p, C.c_int32,
                                    C.POINTER(C.c_int32), C.c_int32, C.c_bool]),
        "sp_tok_embd_rows": (C.c_int32, [C.c_void_p, C.POINTER(C.c_int32),
                                         C.c_int32, C.POINTER(C.c_float)]),
        "sp_decode": (C.c_int32, [C.c_void_p, C.POINTER(C.c_int32), C.c_int32,
                                  C.c_int32, C.c_void_p, C.POINTER(C.c_int32),
                                  C.POINTER(C.c_float), C.c_int32]),
        "sp_logits": (C.POINTER(C.c_float), [C.c_void_p, C.c_int32]),
        "sp_token_to_piece": (C.c_int32, [C.c_void_p, C.c_int32, C.c_char_p, C.c_int32]),
        "sp_is_eog": (C.c_int32, [C.c_void_p, C.c_int32]),
        "sp_reset": (None, [C.c_void_p]),
        "sp_sync": (None, [C.c_void_p]),
    }
    for name, (restype, argtypes) in sig.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes


class Shim:
    """Owns a llama model + context and exposes the tokenizer to Python."""

    def __init__(self, model_path: str, *, n_gpu_layers: int = 0,
                 n_ctx: int = 2048, n_batch: int | None = None,
                 lib_path: str | None = None):
        self._lib = C.CDLL(lib_path or find_shim())
        _bind(self._lib)

        params = ShimParams(
            model_path=str(model_path).encode(),
            n_gpu_layers=int(n_gpu_layers),
            n_ctx=int(n_ctx),
            n_batch=int(n_batch if n_batch is not None else n_ctx),
            n_ubatch=0,
            verbose=0,
        )
        self._h = self._lib.sp_create(C.byref(params))
        if not self._h:
            raise RuntimeError("sp_create returned null")
        if not self._lib.sp_ok(self._h):
            err = self._lib.sp_last_error(self._h).decode()
            self.close()
            raise RuntimeError("sp_create failed: " + err)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.sp_destroy(self._h)
            self._h = None

    def __enter__(self) -> "Shim":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # -- properties --------------------------------------------------------
    @property
    def n_embd(self) -> int:
        return int(self._lib.sp_n_embd(self._h))

    @property
    def n_vocab(self) -> int:
        return int(self._lib.sp_n_vocab(self._h))

    @property
    def n_layer(self) -> int:
        return int(self._lib.sp_n_layer(self._h))

    # -- tokenizer ---------------------------------------------------------
    def tokenize(self, text: str | bytes, *, add_special: bool = False) -> np.ndarray:
        raw = text.encode("utf-8") if isinstance(text, str) else text
        if not raw:
            return np.empty(0, dtype=np.int32)
        need = self._lib.sp_tokenize(self._h, raw, len(raw), None, 0, add_special)
        if need < 0:
            raise RuntimeError("tokenize failed: " +
                               self._lib.sp_last_error(self._h).decode())
        buf = np.empty(need, dtype=np.int32)
        got = self._lib.sp_tokenize(self._h, raw, len(raw),
                                    buf.ctypes.data_as(C.POINTER(C.c_int32)),
                                    need, add_special)
        if got < 0:
            raise RuntimeError("tokenize failed: " +
                               self._lib.sp_last_error(self._h).decode())
        return buf[:got]

    def token_to_piece(self, token: int) -> bytes:
        buf = C.create_string_buffer(256)
        n = self._lib.sp_token_to_piece(self._h, int(token), buf, 256)
        return buf.raw[:n] if n > 0 else b""

    # -- feature extraction ------------------------------------------------
    def set_layers(self, layers) -> None:
        """Arm layer-input extraction. Must run before the first decode()."""
        arr = np.ascontiguousarray(layers, dtype=np.int32)
        rc = self._lib.sp_set_layers(self._h, arr.ctypes.data_as(C.POINTER(C.c_int32)),
                                     arr.size)
        if rc != arr.size:
            raise RuntimeError("set_layers failed: " + self._lib.sp_last_error(self._h).decode())

    @property
    def row_stride(self) -> int:
        return int(self._lib.sp_row_stride(self._h))

    def decode(self, tokens: np.ndarray, start_pos: int = 0,
               feat_ptr: int | None = None, want_logits: bool = True) -> np.ndarray:
        """Decode one batch; features (if feat_ptr) are written straight to memory.

        Returns the per-position argmax token ids.
        """
        toks = np.ascontiguousarray(tokens, dtype=np.int32)
        n = int(toks.size)
        argmax = np.empty(n, dtype=np.int32)
        rc = self._lib.sp_decode(
            self._h,
            toks.ctypes.data_as(C.POINTER(C.c_int32)), n, int(start_pos),
            C.c_void_p(feat_ptr) if feat_ptr else None,
            argmax.ctypes.data_as(C.POINTER(C.c_int32)), None,
            1 if want_logits else 0)
        if rc != n:
            raise RuntimeError("decode failed: " + self._lib.sp_last_error(self._h).decode())
        return argmax

    def reset(self) -> None:
        self._lib.sp_reset(self._h)
