#![crate_name = "powerboxesrs"]

//! # Powerboxesrs
//!
//! Utility functions for transforming bounding boxes and computing metrics.
//!
//! ## Installation
//!
//! ```toml
//! [dependencies]
//! powerboxesrs = "0.3.0"
//! ```
//!
//! ## Slice-based API (no ndarray dependency)
//!
//! All core functions have a `_slice` variant that operates on flat `&[N]` slices,
//! avoiding coupling to a specific `ndarray` version.
//! To opt out of ndarray entirely, disable default features:
//!
//! ```toml
//! [dependencies]
//! powerboxesrs = { version = "0.3", default-features = false }
//! ```
//!
//! ```rust
//! use powerboxesrs::iou::iou_distance_slice;
//!
//! // Flat slice: [x1, y1, x2, y2, ...] with num_boxes
//! let boxes1 = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
//! let boxes2 = vec![0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5];
//! let iou = iou_distance_slice(&boxes1, &boxes2, 2, 2);
//! assert_eq!(iou, vec![0.8571428571428572, 1., 1., 0.8571428571428572]);
//! ```
//!
//! ## ndarray API (default feature)
//!
//! With the `ndarray` feature (enabled by default), wrappers accepting `ArrayView2` are available.
//! The ndarray dependency is flexible (`>=0.15, <=0.16`) to minimize version conflicts.
//!
//! ### Functions available
//! **Note:** all functions expect boxes in `xyxy` format (top left and bottom right corners), except box conversion and rotated box functions.
//!
//! #### Box Transformations and utilities
//! - `box_areas`: Compute the area of list of boxes
//! - `box_convert`: Convert a box from one format to another. Supported formats are `xyxy`, `xywh`, `cxcywh`
//! - `remove_small_boxes`: Remove boxes with area smaller than a threshold
//! - `masks_to_boxes`: Convert a mask to a list of boxes
//!
//! #### Box Metrics
//! - `iou_distance`: Compute the intersection over union matrix of two sets of boxes
//! - `parallel_iou_distance`: Compute the intersection over union matrix of two sets of boxes in parallel
//! - `giou_distance`: Compute the generalized intersection over union matrix of two sets of boxes
//! - `parallel_giou_distance`: Compute the generalized intersection over union matrix of two sets of boxes in parallel
//! - `diou_distance`: Compute the distance intersection over union matrix of two sets of boxes
//! - `tiou_distance`: Compute the tracking intersection over union matrix of two sets of boxes
//!
//! #### Rotated Box Metrics
//! Rotated boxes use `(cx, cy, w, h, angle)` format where angle is in degrees.
//! - `rotated_iou_distance`: Compute IoU distance for rotated boxes
//! - `rotated_giou_distance`: Compute GIoU distance for rotated boxes
//! - `rotated_tiou_distance`: Compute tracking IoU distance for rotated boxes
//!
//! #### Box NMS
//! - `nms`: Non-maximum suppression, returns the indices of the boxes to keep
//! - `rtree_nms`: Non-maximum suppression using an R-tree for sub-quadratic complexity
//!
//! #### Drawing
//! - `draw_boxes`: Draw bounding boxes on a CHW image tensor
//!
pub mod boxes;
pub mod diou;
pub mod draw;
pub mod giou;
pub mod iou;
pub mod nms;
pub mod rotation;
pub mod tiou;
pub(crate) mod utils;
