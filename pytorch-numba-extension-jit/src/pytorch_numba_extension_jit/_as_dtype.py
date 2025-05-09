from __future__ import annotations

import typing

import numpy as np
import torch


class AsDType:
    torch_to_tcpp: typing.Final = {
        torch.float32: "torch::kFloat32",
        torch.uint8: "torch::kUInt8",
        torch.uint16: "torch::kUInt16",
        torch.uint32: "torch::kUInt32",
        torch.uint64: "torch::kUInt64",
        torch.int8: "torch::kInt8",
        torch.int16: "torch::kInt16",
        torch.int32: "torch::kInt32",
        torch.int64: "torch::kInt64",
    }
    torch_to_c: typing.Final = {
        torch.float32: "float",
        torch.uint8: "uint8_t",
        torch.uint16: "uint16_t",
        torch.uint32: "uint32_t",
        torch.uint64: "uint64_t",
        torch.int8: "int8_t",
        torch.int16: "int16_t",
        torch.int32: "int32_t",
        torch.int64: "int64_t",
    }
    torch_to_numba: typing.Final = {
        torch.float32: "float32",
        torch.uint8: "uint8",
        torch.uint16: "uint16",
        torch.uint32: "uint32",
        torch.uint64: "uint64",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
    }
    torch_to_rs: typing.Final = {
        torch.float32: "f32",
        torch.uint8: "u8",
        torch.uint16: "u16",
        torch.uint32: "u32",
        torch.uint64: "u64",
        torch.int8: "i8",
        torch.int16: "i16",
        torch.int32: "i32",
        torch.int64: "i64",
    }
    torch_to_np: typing.Final = {
        torch.float32: np.float32,
        torch.uint8: np.uint8,
        torch.uint16: np.uint16,
        torch.uint32: np.uint32,
        torch.uint64: np.uint64,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
    }
    any_to_torch: typing.Final = (
        {int: torch.int64, float: torch.float32}
        | {k: k for k in torch_to_c}
        | {v: k for k, v in torch_to_tcpp.items()}
        | {v: k for k, v in torch_to_c.items()}
        | {v: k for k, v in torch_to_rs.items()}
        | {v.removesuffix("_t"): k for k, v in torch_to_c.items()}
        | {v: k for k, v in torch_to_np.items()}
        | {v: k for k, v in torch_to_numba.items()}
    )

    def __init__(self, dtype: torch.dtype | np.dtype | str):
        if dtype not in self.any_to_torch:
            raise ValueError(f"Unsupported {dtype=}")
        self.dtype = self.any_to_torch[dtype]

    def byte_width(self):
        try:
            return torch.finfo(self.dtype).bits // 8
        except TypeError:
            return torch.iinfo(self.dtype).bits // 8

    def as_tcpp(self):
        if self.dtype in self.torch_to_tcpp:
            return self.torch_to_tcpp[self.dtype]
        raise ValueError(f"Unsupported dtype {self.dtype}")

    def as_c(self):
        if self.dtype in self.torch_to_c:
            return self.torch_to_c[self.dtype]
        raise ValueError(f"Unsupported dtype {self.dtype}")

    def as_numba(self):
        if self.dtype in self.torch_to_numba:
            return self.torch_to_numba[self.dtype]
        raise ValueError(f"Unsupported dtype {self.dtype}")

    def as_torchlib_scalar(self):
        return "float" if self.dtype.is_floating_point else "int"
