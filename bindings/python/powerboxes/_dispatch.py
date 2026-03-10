"""Build dtype-dispatch dicts from the typed symbols in _powerboxes."""

import numpy as np

from . import _powerboxes as _ext

_ALL_SUFFIXES = ("f64", "f32", "i64", "i32", "i16", "u64", "u32", "u16", "u8")
_SIGNED_SUFFIXES = ("f64", "f32", "i64", "i32", "i16")

_SUFFIX_TO_DTYPE = {
    "f64": np.dtype("float64"),
    "f32": np.dtype("float32"),
    "i64": np.dtype("int64"),
    "i32": np.dtype("int32"),
    "i16": np.dtype("int16"),
    "u64": np.dtype("uint64"),
    "u32": np.dtype("uint32"),
    "u16": np.dtype("uint16"),
    "u8": np.dtype("uint8"),
}


def _build_dispatch(prefix: str, suffixes=_ALL_SUFFIXES) -> dict:
    return {_SUFFIX_TO_DTYPE[s]: getattr(_ext, f"{prefix}_{s}") for s in suffixes}


_dtype_to_func_iou_distance = _build_dispatch("iou_distance")
_dtype_to_func_parallel_iou_distance = _build_dispatch("parallel_iou_distance")
_dtype_to_func_diou_distance = _build_dispatch("diou_distance", ("f64", "f32"))
_dtype_to_func_giou_distance = _build_dispatch("giou_distance")
_dtype_to_func_parallel_giou_distance = _build_dispatch("parallel_giou_distance")
_dtype_to_func_tiou_distance = _build_dispatch("tiou_distance")
_dtype_to_func_box_areas = _build_dispatch("box_areas")
_dtype_to_func_box_convert = _build_dispatch("box_convert")
_dtype_to_func_remove_small_boxes = _build_dispatch("remove_small_boxes")
_dtype_to_func_nms = _build_dispatch("nms")
_dtype_to_func_rotated_nms = _build_dispatch("rotated_nms")
_dtype_to_func_rtree_nms = _build_dispatch("rtree_nms", _SIGNED_SUFFIXES)
_dtype_to_func_rtree_rotated_nms = _build_dispatch(
    "rtree_rotated_nms", _SIGNED_SUFFIXES
)
