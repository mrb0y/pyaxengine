# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

from enum import Enum
import ml_dtypes as mldt
import numpy as np


class VNPUType(Enum):
    DISABLED = 0
    ENABLED = 1
    BIG_LITTLE = 2
    LITTLE_BIG = 3


class ModelType(Enum):
    HALF = 0  # for MC20E, which means chip is AX630C(x), or AX620Q(x)
    FULL = 1  # for MC20E
    SINGLE = 0  # for MC50, which means chip is AX650A or AX650N, and M57H
    DUAL = 1  # for MC50
    TRIPLE = 2  # for MC50


class ChipType(Enum):
    MC20E = 0
    MC50 = 1
    M57H = 2


def get_data_type(engine_type):
    if engine_type == ChipType.MC20E:
        return ModelType.HALF
    elif engine_type == ChipType.MC50:
        return ModelType.SINGLE
    elif engine_type == ChipType.M57H:
        return ModelType.SINGLE
    else:
        raise ValueError("Invalid engine type: %s" % engine_type)


def _transform_dtype(ffi, lib, dtype):
    if dtype == ffi.cast("AX_ENGINE_DATA_TYPE_T", lib.AX_ENGINE_DT_UINT8):
        return np.dtype(np.uint8)
    elif dtype == ffi.cast("AX_ENGINE_DATA_TYPE_T", lib.AX_ENGINE_DT_SINT8):
        return np.dtype(np.int8)
    elif dtype == ffi.cast("AX_ENGINE_DATA_TYPE_T", lib.AX_ENGINE_DT_UINT16):
        return np.dtype(np.uint16)
    elif dtype == ffi.cast("AX_ENGINE_DATA_TYPE_T", lib.AX_ENGINE_DT_SINT16):
        return np.dtype(np.int16)
    elif dtype == ffi.cast("AX_ENGINE_DATA_TYPE_T", lib.AX_ENGINE_DT_UINT32):
        return np.dtype(np.uint32)
    elif dtype == ffi.cast("AX_ENGINE_DATA_TYPE_T", lib.AX_ENGINE_DT_SINT32):
        return np.dtype(np.int32)
    elif dtype == ffi.cast("AX_ENGINE_DATA_TYPE_T", lib.AX_ENGINE_DT_FLOAT32):
        return np.dtype(np.float32)
    elif dtype == ffi.cast("AX_ENGINE_DATA_TYPE_T", lib.AX_ENGINE_DT_BFLOAT16):
        return np.dtype(mldt.bfloat16)
    else:
        raise ValueError(f"Unsupported data type '{dtype}'.")
