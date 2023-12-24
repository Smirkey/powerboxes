[![CI](https://github.com/smirkey/powerboxes/actions/workflows/ci.yml/badge.svg)](https://github.com/smirkey/powerboxes/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/smirkey/powerboxes/branch/main/graph/badge.svg)](https://codecov.io/gh/smirkey/powerboxes)
[![Crates.io](https://img.shields.io/crates/v/powerboxesrs.svg)](https://crates.io/crates/powerboxesrs)

# <div align="center"> PowerBoxesrs </div>
Powerboxesrs is a rust package containing utility functions for transforming bounding boxes and computing metric.

## Installation

```bash
cargo add powerboxesrs
```

### Functions available
#### Box Transformations and utilities
- `box_areas`: Compute the area of list of boxes
- `box_convert`: Convert a box from one format to another. Supported formats are `xyxy`, `xywh`, `cxcywh`.
- `remove_small_boxes`: Remove boxes with area smaller than a threshold
- `mask_to_boxes`: Convert a mask to a list of boxes

#### Box Metrics
- `iou_distance`: Compute the intersection over union matrix of two sets of boxes
- `parallel_iou_distance`: Compute the intersection over union matrix of two sets of boxes in parallel
- `giou_distance`: Compute the generalized intersection over union matrix of two sets of boxes
- `parallel_giou_distance`: Compute the generalized intersection over union matrix of two sets of boxes in parallel

#### Box NMS
- `nms`: Non-maximum suppression, returns the indices of the boxes to keep


## Use it in Rust
See the [documentation](https://docs.rs/powerboxesrs) for more details.
Here is a simple example:
```rust
use ndarray::array;
use powerboxesrs::boxes::box_areas;
let boxes = array![[1., 2., 3., 4.], [0., 0., 10., 10.]];
let areas = box_areas(&boxes);
assert_eq!(areas, array![4., 100.]);
```

