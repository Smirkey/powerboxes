import numpy as np
import pytest
from powerboxes import draw_boxes


class TestDrawBoxes:
    def test_basic_draw(self):
        """Draw a single box on a blank image and verify pixels are set."""
        h, w = 100, 100
        image = np.zeros((3, h, w), dtype=np.uint8)
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
        result = draw_boxes(image, boxes)
        assert result.shape == (3, h, w)
        assert result.dtype == np.uint8
        # Top edge pixel should be colored (default red)
        assert result[0, 10, 25] == 255  # R
        assert result[1, 10, 25] == 0    # G
        assert result[2, 10, 25] == 0    # B
        # Interior pixel should remain black
        assert result[0, 30, 30] == 0

    def test_multiple_boxes_different_colors(self):
        """Multiple boxes get different default colors."""
        h, w = 100, 100
        image = np.zeros((3, h, w), dtype=np.uint8)
        boxes = np.array([
            [5.0, 5.0, 30.0, 30.0],
            [50.0, 50.0, 90.0, 90.0],
        ])
        result = draw_boxes(image, boxes)
        # First box: default red (255, 0, 0)
        assert result[0, 5, 15] == 255
        assert result[1, 5, 15] == 0
        # Second box: default green (0, 255, 0)
        assert result[0, 50, 70] == 0
        assert result[1, 50, 70] == 255

    def test_custom_colors(self):
        """Custom colors are applied correctly."""
        h, w = 50, 50
        image = np.zeros((3, h, w), dtype=np.uint8)
        boxes = np.array([[5.0, 5.0, 40.0, 40.0]])
        colors = np.array([[0, 128, 255]], dtype=np.uint8)
        result = draw_boxes(image, boxes, colors=colors)
        # Top edge should have custom color
        assert result[0, 5, 20] == 0     # R
        assert result[1, 5, 20] == 128   # G
        assert result[2, 5, 20] == 255   # B

    def test_thickness(self):
        """Thicker lines draw more pixels."""
        h, w = 100, 100
        image = np.zeros((3, h, w), dtype=np.uint8)
        boxes = np.array([[20.0, 20.0, 80.0, 80.0]])
        result_thin = draw_boxes(image, boxes, thickness=1)
        result_thick = draw_boxes(image, boxes, thickness=5)
        # Thick drawing should have more non-zero pixels
        assert np.count_nonzero(result_thick) > np.count_nonzero(result_thin)

    def test_empty_boxes(self):
        """No boxes means output equals input."""
        h, w = 50, 50
        image = np.random.randint(0, 256, (3, h, w), dtype=np.uint8)
        boxes = np.zeros((0, 4), dtype=np.float64)
        result = draw_boxes(image, boxes)
        np.testing.assert_array_equal(result, image)

    def test_clipping(self):
        """Boxes extending beyond image boundaries don't crash."""
        h, w = 50, 50
        image = np.zeros((3, h, w), dtype=np.uint8)
        boxes = np.array([[-10.0, -10.0, 60.0, 60.0]])
        result = draw_boxes(image, boxes)
        assert result.shape == (3, h, w)

    def test_preserves_existing_content(self):
        """Drawing on a non-blank image preserves non-box pixels."""
        h, w = 50, 50
        image = np.full((3, h, w), 100, dtype=np.uint8)
        boxes = np.array([[10.0, 10.0, 40.0, 40.0]])
        result = draw_boxes(image, boxes)
        # Interior pixel should still be 100
        assert result[0, 25, 25] == 100

    def test_bad_image_shape(self):
        """Wrong image shape raises ValueError."""
        image = np.zeros((4, 50, 50), dtype=np.uint8)  # 4 channels
        boxes = np.array([[10.0, 10.0, 40.0, 40.0]])
        with pytest.raises(ValueError, match="image must have shape"):
            draw_boxes(image, boxes)

    def test_bad_boxes_shape(self):
        """Wrong boxes shape raises ValueError."""
        image = np.zeros((3, 50, 50), dtype=np.uint8)
        boxes = np.array([[10.0, 10.0, 40.0]])  # only 3 cols
        with pytest.raises(ValueError, match="boxes must have shape"):
            draw_boxes(image, boxes)

    def test_bad_input_types(self):
        """Non-numpy inputs raise TypeError."""
        with pytest.raises(TypeError):
            draw_boxes([[0]], [[10, 10, 40, 40]])

    def test_output_is_copy(self):
        """draw_boxes returns a new array, doesn't mutate input."""
        h, w = 50, 50
        image = np.zeros((3, h, w), dtype=np.uint8)
        boxes = np.array([[10.0, 10.0, 40.0, 40.0]])
        result = draw_boxes(image, boxes)
        assert not np.array_equal(result, image) or np.all(image == 0)
        # Original should still be all zeros
        assert np.all(image == 0)
