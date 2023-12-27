import numpy as np

from ._powerboxes import (
    nms_f32,
    nms_f64,
    nms_i16,
    nms_i32,
    nms_i64,
    nms_u8,
    nms_u16,
    nms_u32,
    nms_u64,
    rtree_nms_f32,
    rtree_nms_f64,
    rtree_nms_i16,
    rtree_nms_i32,
    rtree_nms_i64,
)

_dtype_to_func_nms = {
    np.dtype("float64"): nms_f64,
    np.dtype("float32"): nms_f32,
    np.dtype("int64"): nms_i64,
    np.dtype("int32"): nms_i32,
    np.dtype("int16"): nms_i16,
    np.dtype("uint64"): nms_u64,
    np.dtype("uint32"): nms_u32,
    np.dtype("uint16"): nms_u16,
    np.dtype("uint8"): nms_u8,
}

_dtype_to_func_rtree_nms = {
    np.dtype("float64"): rtree_nms_f64,
    np.dtype("float32"): rtree_nms_f32,
    np.dtype("int64"): rtree_nms_i64,
    np.dtype("int32"): rtree_nms_i32,
    np.dtype("int16"): rtree_nms_i16,
}
