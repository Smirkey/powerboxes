import numpy as np

from .powerboxesrs import (
    box_areas_f32,
    box_areas_f64,
    box_areas_i16,
    box_areas_i32,
    box_areas_i64,
    box_areas_u8,
    box_areas_u16,
    box_areas_u32,
    box_areas_u64,
    box_convert_f32,
    box_convert_f64,
    box_convert_i16,
    box_convert_i32,
    box_convert_i64,
    box_convert_u8,
    box_convert_u16,
    box_convert_u32,
    box_convert_u64,
    remove_small_boxes_f32,
    remove_small_boxes_f64,
    remove_small_boxes_i16,
    remove_small_boxes_i32,
    remove_small_boxes_i64,
    remove_small_boxes_u8,
    remove_small_boxes_u16,
    remove_small_boxes_u32,
    remove_small_boxes_u64,
)

_dtype_to_func_box_areas = {
    np.dtype("float64"): box_areas_f64,
    np.dtype("float32"): box_areas_f32,
    np.dtype("int64"): box_areas_i64,
    np.dtype("int32"): box_areas_i32,
    np.dtype("int16"): box_areas_i16,
    np.dtype("uint64"): box_areas_u64,
    np.dtype("uint32"): box_areas_u32,
    np.dtype("uint16"): box_areas_u16,
    np.dtype("uint8"): box_areas_u8,
}

_dtype_to_func_box_convert = {
    np.dtype("float64"): box_convert_f64,
    np.dtype("float32"): box_convert_f32,
    np.dtype("int64"): box_convert_i64,
    np.dtype("int32"): box_convert_i32,
    np.dtype("int16"): box_convert_i16,
    np.dtype("uint64"): box_convert_u64,
    np.dtype("uint32"): box_convert_u32,
    np.dtype("uint16"): box_convert_u16,
    np.dtype("uint8"): box_convert_u8,
}

_dtype_to_func_remove_small_boxes = {
    np.dtype("float64"): remove_small_boxes_f64,
    np.dtype("float32"): remove_small_boxes_f32,
    np.dtype("int64"): remove_small_boxes_i64,
    np.dtype("int32"): remove_small_boxes_i32,
    np.dtype("int16"): remove_small_boxes_i16,
    np.dtype("uint64"): remove_small_boxes_u64,
    np.dtype("uint32"): remove_small_boxes_u32,
    np.dtype("uint16"): remove_small_boxes_u16,
    np.dtype("uint8"): remove_small_boxes_u8,
}
