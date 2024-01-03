import numpy as np

from ._powerboxes import (
    tiou_distance_f32,
    tiou_distance_f64,
    tiou_distance_i16,
    tiou_distance_i32,
    tiou_distance_i64,
    tiou_distance_u8,
    tiou_distance_u16,
    tiou_distance_u32,
    tiou_distance_u64,
)

_dtype_to_func_tiou_distance = {
    np.dtype("float64"): tiou_distance_f64,
    np.dtype("float32"): tiou_distance_f32,
    np.dtype("int64"): tiou_distance_i64,
    np.dtype("int32"): tiou_distance_i32,
    np.dtype("int16"): tiou_distance_i16,
    np.dtype("uint64"): tiou_distance_u64,
    np.dtype("uint32"): tiou_distance_u32,
    np.dtype("uint16"): tiou_distance_u16,
    np.dtype("uint8"): tiou_distance_u8,
}
