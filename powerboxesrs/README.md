[![CI](https://github.com/smirkey/powerboxes/actions/workflows/ci.yml/badge.svg)](https://github.com/smirkey/powerboxes/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/smirkey/powerboxes/branch/main/graph/badge.svg)](https://codecov.io/gh/smirkey/powerboxes)
[![Crates.io](https://img.shields.io/crates/v/powerboxesrs.svg)](https://crates.io/crates/powerboxesrs)

# <div align="center"> PowerBoxesrs </div>
Powerboxes is a package containing utility functions for transforming bounding boxes and computing metrics.

## Installation

```bash
cargo add powerboxesrs
```

### Functions available

**Note:** all functions expect boxes in `xyxy` format (top left and bottom right corners), except box conversion and rotated box functions.

All core functions have a `_slice` variant operating on flat `&[N]` slices. ndarray wrappers are available behind the `ndarray` feature (enabled by default).

#### Box Transformations and utilities
- `box_areas`: Compute the area of list of boxes
- `box_convert`: Convert a box from one format to another. Supported formats are `xyxy`, `xywh`, `cxcywh`
- `remove_small_boxes`: Remove boxes with area smaller than a threshold
- `masks_to_boxes`: Convert a mask to a list of boxes

#### Box Metrics
- `iou_distance`: Compute the intersection over union matrix of two sets of boxes
- `parallel_iou_distance`: Compute the intersection over union matrix of two sets of boxes in parallel
- `giou_distance`: Compute the generalized intersection over union matrix of two sets of boxes
- `parallel_giou_distance`: Compute the generalized intersection over union matrix of two sets of boxes in parallel
- `diou_distance`: Compute the distance intersection over union matrix of two sets of boxes
- `tiou_distance`: Compute the tracking intersection over union matrix of two sets of boxes

#### Rotated Box Metrics
Rotated boxes use `(cx, cy, w, h, angle)` format where angle is in degrees.
- `rotated_iou_distance`: Compute IoU distance for rotated boxes
- `rotated_giou_distance`: Compute GIoU distance for rotated boxes
- `rotated_tiou_distance`: Compute tracking IoU distance for rotated boxes

#### Box NMS
- `nms`: Non-maximum suppression, returns the indices of the boxes to keep
- `rtree_nms`: Non-maximum suppression using an R-tree for sub-quadratic complexity

#### Drawing
- `draw_boxes`: Draw bounding boxes on a CHW image tensor


## Usage

See the [documentation](https://docs.rs/powerboxesrs) for more details.

### Slice-based API (no ndarray dependency)

All core functions have a `_slice` variant that operates on flat `&[N]` slices. This avoids coupling to a specific `ndarray` version ([#60](https://github.com/Smirkey/powerboxes/issues/60)).

To use powerboxesrs without ndarray:
```toml
[dependencies]
powerboxesrs = { version = "0.3", default-features = false }
```

```rust
use powerboxesrs::iou::iou_distance_slice;

// Flat slice: [x1, y1, x2, y2, ...] with num_boxes
let boxes1 = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
let boxes2 = vec![0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5];
let iou = iou_distance_slice(&boxes1, &boxes2, 2, 2);
```

### ndarray API (default)

With the `ndarray` feature (enabled by default), you get wrappers that accept `ArrayView2` directly. The ndarray dependency is flexible (`>=0.15, <=0.16`) to minimize version conflicts.

```rust
use powerboxesrs::iou::iou_distance;
use ndarray::array;

let boxes1 = array![[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]];
let boxes2 = array![[0.5, 0.5, 1.5, 1.5], [2.5, 2.5, 3.5, 3.5]];
let iou = iou_distance(boxes1.view(), boxes2.view());
```
