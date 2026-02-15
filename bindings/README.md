[![CI](https://github.com/smirkey/powerboxes/actions/workflows/ci.yml/badge.svg)](https://github.com/smirkey/powerboxes/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/smirkey/powerboxes/branch/main/graph/badge.svg)](https://codecov.io/gh/smirkey/powerboxes)
[![pypi](https://img.shields.io/pypi/v/powerboxes.svg)](https://pypi.python.org/pypi/powerboxes)
[![versions](https://img.shields.io/pypi/pyversions/powerboxes.svg)](https://github.com/smirkey/powerboxes)

# PowerBoxes

Powerboxes is a package containing utility functions for transforming bounding boxes and computing metrics. It is implemented in both Python and Rust.
It shows a significant speedup over the equivalent numpy implementations in Python, or other libraries such as [shapely](https://github.com/shapely/shapely) or [torchvision](https://pytorch.org/vision/main/ops.html).

**[Documentation](https://smirkey.github.io/powerboxes/)**

## Installation

```bash
pip install powerboxes
```

## Usage

```python
import powerboxes as pb
import numpy as np

boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.float64)

# Compute areas
areas = pb.box_areas(boxes)

# Compute IoU distance matrix
iou = pb.iou_distance(boxes, boxes)

# Draw boxes on an image (CHW format, uint8)
image = np.zeros((3, 100, 100), dtype=np.uint8)
draw_boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
result = pb.draw_boxes(image, draw_boxes)
```

For more details, see the [full documentation](https://smirkey.github.io/powerboxes/) or the [GitHub repository](https://github.com/Smirkey/powerboxes).
