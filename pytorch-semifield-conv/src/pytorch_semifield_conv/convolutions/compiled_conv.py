from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeVar

import torch
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin

from .utils import ConvMeta

R = TypeVar("R")


class _SemifieldCompiler(Protocol):
    def compile(
        self,
        meta: ConvMeta,
        thread_block_size: int = 256,
        debug: bool = False,
        to_extension: bool = True,
    ) -> Callable[[torch.Tensor, torch.Tensor], R]: ...

    def get_result(self, val: R) -> torch.Tensor: ...


class CompiledConv(nn.Module):
    def __init__(
        self,
        semifield: _SemifieldCompiler,
        thread_block_size: int = 256,
        debug: bool = False,
        to_extension: bool = True,
    ):
        super().__init__()
        self.semifield = semifield
        self.op: Callable[[torch.Tensor, torch.Tensor], Any] | None = None
        self.meta: ConvMeta | None = None
        self.thread_block_size = thread_block_size
        self.debug = debug
        self.to_extension = to_extension

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
            self.op = self.semifield.compile(
                self.meta,
                thread_block_size=self.thread_block_size,
                debug=self.debug,
                to_extension=self.to_extension,
            )

        res = self.op(img, kernel)
        return self.semifield.get_result(res)


class CompiledConvFixed(nn.Module):
    op: Callable[[torch.Tensor, torch.Tensor], Any] | None
    meta: ConvMeta | None
    semifield: _SemifieldCompiler

    def forward(
        self,
        imgs: torch.Tensor,
        kernel: torch.Tensor,
        *args,
        debug: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if debug:
            if not self.meta.check_matches(
                tuple(imgs.shape), tuple(kernel.shape), *args, **kwargs
            ):
                raise ValueError("Failed to match arguments!")
            if self.op is None:
                raise ValueError("Operator not initialised!")

        res = self.op(imgs, kernel)
        return self.semifield.get_result(res)


class CompiledConvFixedLazy(LazyModuleMixin, CompiledConvFixed):
    cls_to_become = CompiledConvFixed

    def __init__(
        self,
        semifield: _SemifieldCompiler,
        thread_block_size: int = 256,
        debug: bool = False,
        to_extension: bool = True,
    ):
        super().__init__()
        self.semifield = semifield
        self.op = None
        self.meta = None
        self.done = False
        self.thread_block_size = thread_block_size
        self.debug = debug
        self.to_extension = to_extension

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
        self.op = self.semifield.compile(
            self.meta,
            thread_block_size=self.thread_block_size,
            debug=self.debug,
            to_extension=self.to_extension,
        )
        self.done = True

    def has_uninitialized_params(self):
        return self.op is None
