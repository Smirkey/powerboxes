
[![CI](https://github.com/smirkey/powerboxes/actions/workflows/ci.yml/badge.svg)](https://github.com/smirkey/powerboxes/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/smirkey/powerboxes/branch/main/graph/badge.svg)](https://codecov.io/gh/smirkey/powerboxes)
[![Crates.io](https://img.shields.io/crates/v/powerboxesrs.svg)](https://crates.io/crates/powerboxesrs)
[![pypi](https://img.shields.io/pypi/v/powerboxes.svg)](https://pypi.python.org/pypi/powerboxes)
[![versions](https://img.shields.io/pypi/pyversions/powerboxes.svg)](https://github.com/smirkey/powerboxes)

# <div align="center"> PowerBoxes </div>
Powerboxes is a package containing utility functions for transforming bounding boxes and computing metrics. It is implemented in both Python and Rust.
It shows a significant speedup over the equivalent numpy implementations in Python, or other libraries such as [shapely](https://github.com/shapely/shapely) or [torchvision](https://pytorch.org/vision/main/ops.html).


See source code [here](https://github.com/Smirkey/powerboxes)

### Installation
```console
pip install powerboxes
```

## Example Usage

```python
import powerboxes as pb
import numpy as np

# Create bounding boxes in xyxy format
boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.float64)

# Compute areas
areas = pb.boxes_areas(boxes)

# Compute pairwise IoU distance matrix
iou = pb.iou_distance(boxes, boxes)

# Non-maximum suppression
scores = np.array([0.9, 0.8])
keep = pb.nms(boxes, scores, iou_threshold=0.5, score_threshold=0.3)

# Draw boxes on an image (CHW format, uint8)
image = np.zeros((3, 100, 100), dtype=np.uint8)
draw_boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
result = pb.draw_boxes(image, draw_boxes)
```

!!! note "supported dtypes by most functions"

    ::: powerboxes.supported_dtypes
