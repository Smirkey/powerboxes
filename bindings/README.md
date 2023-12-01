[![CI](https://github.com/smirkey/powerboxes/actions/workflows/ci.yml/badge.svg)](https://github.com/smirkey/powerboxes/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/smirkey/powerboxes/branch/main/graph/badge.svg)](https://codecov.io/gh/smirkey/powerboxes)
[![pypi](https://img.shields.io/pypi/v/powerboxes.svg)](https://pypi.python.org/pypi/powerboxes)
[![versions](https://img.shields.io/pypi/pyversions/powerboxes.svg)](https://github.com/smirkey/powerboxes)

# <div align="center"> PowerBoxes </div>
Powerboxes is a package containing utility functions for transforming bounding boxes and computing metrics. It is implemented in Rust with python bindings.
It shows a significant speedup over the equivalent numpy implementations in Python, or other libraries such as [shapely](https://github.com/shapely/shapely).

## Installation

```bash
pip install powerboxes
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

#### Box Metrics
- `iou_distance`: Compute the intersection over union matrix of two sets of boxes
- `parallel_iou_distance`: Compute the intersection over union matrix of two sets of boxes in parallel
- `giou_distance`: Compute the generalized intersection over union matrix of two sets of boxes
- `parallel_giou_distance`: Compute the generalized intersection over union matrix of two sets of boxes in parallel
