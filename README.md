[![CI](https://github.com/smirkey/powerboxes/actions/workflows/ci.yml/badge.svg)](https://github.com/smirkey/powerboxes/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/smirkey/powerboxes/branch/main/graph/badge.svg)](https://codecov.io/gh/smirkey/powerboxes)
[![Crates.io](https://img.shields.io/crates/v/powerboxesrs.svg)](https://crates.io/crates/powerboxesrs)
[![pypi](https://img.shields.io/pypi/v/powerboxes.svg)](https://pypi.python.org/pypi/powerboxes)
[![versions](https://img.shields.io/pypi/pyversions/powerboxes.svg)](https://github.com/smirkey/powerboxes)

# <div align="center"> PowerBoxes </div>
Powerboxes is a package containing utility functions for transforming bounding boxes and computing metrics. It is implemented in both Python and Rust.
It shows a significant speedup over the equivalent numpy implementations in Python, or other libraries such as [shapely](https://github.com/shapely/shapely) or [torchvision](https://pytorch.org/vision/main/ops.html).

## Installation

### Python
```bash
pip install powerboxes
```

### Rust
```bash
cargo add powerboxesrs
```

## Python Usage

```python
import powerboxes as pb
import numpy as np

# Create a bounding box
box = np.array([[0, 0, 1, 1]])

# Compute the area of the box
area = pb.box_areas(box)

# Compute the intersection of the box with itself
intersection = pb.iou_distance(box, box)
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
- `rtree_nms`: Non-maximum suppression, returns the indices of the boxes to keep, uses a r-tree internally to avoid quadratic complexity, useful when having many boxes.

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


## Benchmarks

Some benchmarks of powerboxes against various open source alternatives, not all functions are benchmarked. Notice that we use log scales, **all differences are major** !
Benchmarks can be found in this google colab [notebook](https://colab.research.google.com/drive/1Z8auT4GZFbwaNs9hZfnB0kvYBbX-MOgS?usp=sharing)

### Box area
Here it's torchvision vs powerboxes vs numpy

![Box area](./images/box_area.png)

### Box convert
Here it's torchvision vs powerboxes

![Box convert](./images/box_convert.png)

### Box IoU matrix
Torchvision vs numpy vs powerboxes

![Box IoU](./images/box_iou.png)

### NMS
Torchvision vs powerboxes vs lsnms vs numpy

#### Large image (10000x10000 pixels)

![Box NMS](./images/box_nms_large_image.png)

#### Normal image (1000x1000 pixels)

![Box NMS](./images/box_nms_normal_image.png)