"""Epilogue-specific GEMM autotune policy.

The class is intentionally a thin leaf: variable-length identity and selector
behaviour live in :class:`DLKernel.varlen_selector.VarlenGemmTunePolicy`, while
this module contributes only explicit epilogue tensor roles and host legality.
"""

from __future__ import annotations

from typing import Iterable, Optional

from DLKernel.varlen_selector import VarlenGemmTunePolicy, gemm_tune_key
from DLKernel.epilogue.legality import prune_epilogue_configs


class GemmEpilogueTunePolicy(VarlenGemmTunePolicy):
    """Add epilogue/mod constraints on top of the varlen GEMM policy."""

    def __init__(
        self,
        mod,
        transform_a=None,
        *,
        ragged_m_out_names: Optional[Iterable[str]] = None,
        opaque_tensor_names: Iterable[str] = (),
    ) -> None:
        super().__init__()
        self.mod = mod
        self.transform_a = transform_a
        sink_names = tuple(
            name for name, op in mod.sinks.items() if hasattr(op, "sink_alloc_shape")
        )
        declared_outputs = tuple(mod.outputs)
        if ragged_m_out_names is None:
            ragged_m_out_names = ("out", "C", "D", *declared_outputs)
        self._ragged_m_out_names = frozenset(ragged_m_out_names)
        self._opaque_tensor_names = frozenset((*sink_names, *opaque_tensor_names))

    def bind_signature(self, arg_names, arg_defaults) -> None:
        super().bind_signature(arg_names, arg_defaults)
        self._opaque_tensor_names = frozenset(
            (
                *self._opaque_tensor_names,
                *(name for name in self.arg_names if name.startswith("ta__")),
            )
        )

    def make_key(self, args, kwargs, default_key):
        if not getattr(self, "_varlen_params", False):
            return default_key()
        key = gemm_tune_key(
            args,
            kwargs,
            self.arg_names,
            self.arg_defaults,
            ragged_m_out_names=self._ragged_m_out_names,
            opaque_tensor_names=self._opaque_tensor_names,
        )
        return default_key() if key is None else key

    def prepare_candidates(self, configs, named_args, kwargs):
        configs = super().prepare_candidates(configs, named_args, kwargs)
        return prune_epilogue_configs(self.mod, self.transform_a, configs, named_args, **kwargs)


__all__ = ["GemmEpilogueTunePolicy"]
