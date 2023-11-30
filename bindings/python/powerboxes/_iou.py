import numpy as np

from ._powerboxes import (
    iou_distance_f32,
    iou_distance_f64,
    iou_distance_i16,
    iou_distance_i32,
    iou_distance_i64,
    iou_distance_u8,
    iou_distance_u16,
    iou_distance_u32,
    iou_distance_u64,
    parallel_iou_distance_f32,
    parallel_iou_distance_f64,
    parallel_iou_distance_i16,
    parallel_iou_distance_i32,
    parallel_iou_distance_i64,
    parallel_iou_distance_u8,
    parallel_iou_distance_u16,
    parallel_iou_distance_u32,
    parallel_iou_distance_u64,
)

_dtype_to_func_parallel_iou_distance = {
    np.dtype("float64"): parallel_iou_distance_f64,
    np.dtype("float32"): parallel_iou_distance_f32,
    np.dtype("int64"): parallel_iou_distance_i64,
    np.dtype("int32"): parallel_iou_distance_i32,
    np.dtype("int16"): parallel_iou_distance_i16,
    np.dtype("uint64"): parallel_iou_distance_u64,
    np.dtype("uint32"): parallel_iou_distance_u32,
    np.dtype("uint16"): parallel_iou_distance_u16,
    np.dtype("uint8"): parallel_iou_distance_u8,
}
_dtype_to_func_iou_distance = {
    np.dtype("float64"): iou_distance_f64,
    np.dtype("float32"): iou_distance_f32,
    np.dtype("int64"): iou_distance_i64,
    np.dtype("int32"): iou_distance_i32,
    np.dtype("int16"): iou_distance_i16,
    np.dtype("uint64"): iou_distance_u64,
    np.dtype("uint32"): iou_distance_u32,
    np.dtype("uint16"): iou_distance_u16,
    np.dtype("uint8"): iou_distance_u8,
}
