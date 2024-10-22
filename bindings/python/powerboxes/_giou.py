# import numpy as np

# from ._powerboxes import (
#     giou_distance_f32,
#     giou_distance_f64,
#     giou_distance_i16,
#     giou_distance_i32,
#     giou_distance_i64,
#     giou_distance_u8,
#     giou_distance_u16,
#     giou_distance_u32,
#     giou_distance_u64,
#     parallel_giou_distance_f32,
#     parallel_giou_distance_f64,
#     parallel_giou_distance_i16,
#     parallel_giou_distance_i32,
#     parallel_giou_distance_i64,
#     parallel_giou_distance_u8,
#     parallel_giou_distance_u16,
#     parallel_giou_distance_u32,
#     parallel_giou_distance_u64,
# )

# _dtype_to_func_giou_distance = {
#     np.dtype("float64"): giou_distance_f64,
#     np.dtype("float32"): giou_distance_f32,
#     np.dtype("int64"): giou_distance_i64,
#     np.dtype("int32"): giou_distance_i32,
#     np.dtype("int16"): giou_distance_i16,
#     np.dtype("uint64"): giou_distance_u64,
#     np.dtype("uint32"): giou_distance_u32,
#     np.dtype("uint16"): giou_distance_u16,
#     np.dtype("uint8"): giou_distance_u8,
# }
# _dtype_to_func_parallel_giou_distance = {
#     np.dtype("float64"): parallel_giou_distance_f64,
#     np.dtype("float32"): parallel_giou_distance_f32,
#     np.dtype("int64"): parallel_giou_distance_i64,
#     np.dtype("int32"): parallel_giou_distance_i32,
#     np.dtype("int16"): parallel_giou_distance_i16,
#     np.dtype("uint64"): parallel_giou_distance_u64,
#     np.dtype("uint32"): parallel_giou_distance_u32,
#     np.dtype("uint16"): parallel_giou_distance_u16,
#     np.dtype("uint8"): parallel_giou_distance_u8,
# }
