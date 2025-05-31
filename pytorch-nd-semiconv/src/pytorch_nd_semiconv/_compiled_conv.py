from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol, TypeVar

import torch
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin

from ._utils import ConvMeta

R = TypeVar("R")


class _SemifieldCompiler(Protocol):
    def _compile(
        self,
        meta: ConvMeta,
        compile_options: Mapping[str, Any],
    ) -> Callable[[torch.Tensor, torch.Tensor], R]: ...

    def _get_result(self, val: R) -> torch.Tensor: ...


class FrozenDict(dict):
    def __setitem__(self, key, value):
        raise TypeError("Dict is frozen!")

    def __delitem__(self, key):
        raise TypeError("Dict is frozen!")

    def __hash__(self):
        return hash(tuple(self.items()))


class CompiledConv(nn.Module):
    def __init__(
        self,
        semifield: _SemifieldCompiler,
        compile_options: dict[str, Any],
    ):
        super().__init__()
        self.semifield = semifield
        self.op: Callable[[torch.Tensor, torch.Tensor], Any] | None = None
        self.meta: ConvMeta | None = None
        self.compile_options = FrozenDict(compile_options)

    def forward(
        self,
        img: torch.Tensor,
        kernel: torch.Tensor,
        stride: int | tuple[int, int] = 1,
        padding: (
            int
            | tuple[int, int]
            | tuple[tuple[int, int], tuple[int, int]]
            | Literal["valid", "same"]
        ) = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        kind: Literal["conv", "corr"] = "conv",
    ):
        if self.op is None or not self.meta.check_matches(
            tuple(img.shape),
            tuple(kernel.shape),
            stride,
            padding,
            dilation,
            groups,
            group_broadcasting,
            kind,
        ):
            self.meta = ConvMeta.infer(
                tuple(img.shape),
                tuple(kernel.shape),
                stride,
                padding,
                dilation,
                groups,
                group_broadcasting,
                kind,
            )
            # noinspection PyProtectedMember
            self.op = self.semifield._compile(self.meta, self.compile_options)

        res = self.op(img, kernel)
        # noinspection PyProtectedMember
        return self.semifield._get_result(res)

    def extra_repr(self) -> str:
        return "uninitialised" if self.op is None else "initialised"


class CompiledConvFixed(nn.Module):
    op: Callable[[torch.Tensor, torch.Tensor], Any] | None
    meta: ConvMeta | None
    semifield: _SemifieldCompiler
    debug: bool = False

    def forward(
        self,
        imgs: torch.Tensor,
        kernel: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self.debug:
            if self.op is None:
                raise ValueError("Operator not initialised!")
            if not self.meta.check_matches(
                tuple(imgs.shape), tuple(kernel.shape), *args, **kwargs
            ):
                raise ValueError("Failed to match arguments!")

        res = self.op(imgs, kernel)
        # noinspection PyProtectedMember
        return self.semifield._get_result(res)

    def extra_repr(self) -> str:
        return "INVALID" if self.op is None else "fixed"


class CompiledConvFixedLazy(LazyModuleMixin, CompiledConvFixed):
    cls_to_become = CompiledConvFixed

    def __init__(
        self,
        semifield: _SemifieldCompiler,
        compile_options: dict[str, Any],
    ):
        super().__init__()
        self.semifield = semifield
        self.op = None
        self.meta = None
        self.done = False
        self.compile_options = FrozenDict(compile_options)
        self.debug = compile_options.get("debug", False)

    def initialize_parameters(
        self,
        imgs: torch.Tensor,
        kernel: torch.Tensor,
        stride: int | tuple[int, int] = 1,
        padding: (
            int
            | tuple[int, int]
            | tuple[tuple[int, int], tuple[int, int]]
            | Literal["valid", "same"]
        ) = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        group_broadcasting: bool = False,
        kind: Literal["conv", "corr"] = "conv",
    ):
        assert not self.done
        self.meta = ConvMeta.infer(
            tuple(imgs.shape),
            tuple(kernel.shape),
            stride,
            padding,
            dilation,
            groups,
            group_broadcasting,
            kind,
        )
        # noinspection PyProtectedMember
        self.op = self.semifield._compile(self.meta, self.compile_options)
        self.done = True

    def has_uninitialized_params(self):
        return self.op is None

    def extra_repr(self) -> str:
        return "uninitialised" if self.op is None else "INVALID"
