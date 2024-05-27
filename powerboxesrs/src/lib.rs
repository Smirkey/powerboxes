#![crate_name = "powerboxesrs"]

//! Powerboxes is a package containing utility functions for transforming bounding boxes and computing metrics from them.
//! # Powerboxesrs
//!
//! `powerboxesrs` is a Rust package containing utility functions for transforming bounding boxes and computing metrics from them.
//!
//! ## Installation
//!
//! Add the following to your `Cargo.toml` file:
//!
//! ```toml
//! [dependencies]
//! powerboxesrs = "0.2.3"
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use ndarray::array;
//! use powerboxesrs::iou::iou_distance;
//!
//! let boxes1 = array![[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]];
//! let boxes2 = array![[0.5, 0.5, 1.5, 1.5], [2.5, 2.5, 3.5, 3.5]];
//! let iou = iou_distance(&boxes1, &boxes2);
//! assert_eq!(iou, array![[0.8571428571428572, 1.],[1., 0.8571428571428572]]);
//! ```
//!
//! ### Functions available
//! warning: **all functions expect the boxes to be in the format `xyxy` (top left and bottom right corners)** (not box conversion functions, of course)
//!
//! #### Box Transformations and utilities
//! - `box_areas`: Compute the area of list of boxes
//! - `box_convert`: Convert a box from one format to another. Supported formats are `xyxy`, `xywh`, `cxcywh`.
//! - `remove_small_boxes`: Remove boxes with area smaller than a threshold
//! - `mask_to_boxes`: Convert a mask to a list of boxes
//!
//! #### Box Metrics
//! - `iou_distance`: Compute the intersection over union matrix of two sets of boxes
//! - `parallel_iou_distance`: Compute the intersection over union matrix of two sets of boxes in parallel
//! - `giou_distance`: Compute the generalized intersection over union matrix of two sets of boxes
//! - `parallel_giou_distance`: Compute the generalized intersection over union matrix of two sets of boxes in parallel
//!
//! #### Box NMS
//! - `nms`: Non-maximum suppression, returns the indices of the boxes to keep
//! - `rtree_nms`: Non-maximum suppression, returns the indices of the boxes to keep, uses a r-tree internally to avoid quadratic complexity, useful when having many boxes.
//!
pub mod boxes;
pub mod giou;
pub mod iou;
pub mod nms;
pub mod rotation;
pub mod tiou;
mod utils;
pub mod diou;
