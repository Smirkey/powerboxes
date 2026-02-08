/// Draw bounding boxes on a CHW image tensor.
///
/// # Arguments
/// * `image` - flat slice of length 3*H*W (CHW layout), values 0-255
/// * `height` - image height
/// * `width` - image width
/// * `boxes` - flat slice of length N*4 (xyxy format, f64)
/// * `num_boxes` - number of boxes
/// * `colors` - flat slice of length N*3 (RGB per box), or None for default colors
/// * `thickness` - line thickness in pixels
///
/// # Returns
/// New Vec<u8> of length 3*H*W with boxes drawn
pub fn draw_boxes_slice(
    image: &[u8],
    height: usize,
    width: usize,
    boxes: &[f64],
    num_boxes: usize,
    colors: Option<&[u8]>,
    thickness: usize,
) -> Vec<u8> {
    let mut output = image.to_vec();
    let thickness = if thickness == 0 { 1 } else { thickness };
    let half_t = thickness / 2;

    // Default color palette (cycling)
    let default_colors: [(u8, u8, u8); 10] = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (255, 128, 0),
    ];

    for b in 0..num_boxes {
        let base = b * 4;
        let x1 = boxes[base].round().max(0.0) as usize;
        let y1 = boxes[base + 1].round().max(0.0) as usize;
        let x2 = boxes[base + 2].round().min(width as f64 - 1.0).max(0.0) as usize;
        let y2 = boxes[base + 3].round().min(height as f64 - 1.0).max(0.0) as usize;

        if x1 >= width || y1 >= height {
            continue;
        }

        let (r, g, bl) = if let Some(c) = colors {
            let cb = b * 3;
            (c[cb], c[cb + 1], c[cb + 2])
        } else {
            default_colors[b % default_colors.len()]
        };

        // Draw horizontal lines (top and bottom)
        for x in x1..=x2 {
            // Top edge
            for dy in 0..thickness {
                let y = if y1 + dy >= half_t { y1 + dy - half_t } else { 0 };
                if y < height && x < width {
                    set_pixel_chw(&mut output, height, width, y, x, r, g, bl);
                }
            }
            // Bottom edge
            for dy in 0..thickness {
                let y = if y2 + dy >= half_t { y2 + dy - half_t } else { 0 };
                if y < height && x < width {
                    set_pixel_chw(&mut output, height, width, y, x, r, g, bl);
                }
            }
        }

        // Draw vertical lines (left and right)
        for y in y1..=y2 {
            // Left edge
            for dx in 0..thickness {
                let x = if x1 + dx >= half_t { x1 + dx - half_t } else { 0 };
                if y < height && x < width {
                    set_pixel_chw(&mut output, height, width, y, x, r, g, bl);
                }
            }
            // Right edge
            for dx in 0..thickness {
                let x = if x2 + dx >= half_t { x2 + dx - half_t } else { 0 };
                if y < height && x < width {
                    set_pixel_chw(&mut output, height, width, y, x, r, g, bl);
                }
            }
        }
    }

    output
}

#[inline]
fn set_pixel_chw(image: &mut [u8], height: usize, width: usize, y: usize, x: usize, r: u8, g: u8, b: u8) {
    let hw = height * width;
    image[y * width + x] = r;             // channel 0 (R)
    image[hw + y * width + x] = g;        // channel 1 (G)
    image[2 * hw + y * width + x] = b;    // channel 2 (B)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draw_boxes_on_blank_image() {
        let height = 10;
        let width = 10;
        let image = vec![0u8; 3 * height * width];
        let boxes = vec![2.0, 2.0, 7.0, 7.0];
        let result = draw_boxes_slice(&image, height, width, &boxes, 1, None, 1);

        // Check that the top-left corner pixel (2,2) is set to the first default color (255, 0, 0)
        let hw = height * width;
        assert_eq!(result[2 * width + 2], 255); // R channel
        assert_eq!(result[hw + 2 * width + 2], 0); // G channel
        assert_eq!(result[2 * hw + 2 * width + 2], 0); // B channel

        // Check a pixel on the top edge (2, 4)
        assert_eq!(result[2 * width + 4], 255);

        // Check an interior pixel that should remain black (4, 4)
        assert_eq!(result[4 * width + 4], 0);
    }

    #[test]
    fn test_draw_boxes_with_custom_colors() {
        let height = 10;
        let width = 10;
        let image = vec![0u8; 3 * height * width];
        let boxes = vec![1.0, 1.0, 5.0, 5.0];
        let colors = vec![0u8, 255, 128];
        let result = draw_boxes_slice(&image, height, width, &boxes, 1, Some(&colors), 1);

        let hw = height * width;
        // Check top-left corner of box (1, 1) has custom color
        assert_eq!(result[1 * width + 1], 0);     // R
        assert_eq!(result[hw + 1 * width + 1], 255); // G
        assert_eq!(result[2 * hw + 1 * width + 1], 128); // B
    }

    #[test]
    fn test_draw_boxes_empty() {
        let height = 10;
        let width = 10;
        let image = vec![0u8; 3 * height * width];
        let boxes: Vec<f64> = vec![];
        let result = draw_boxes_slice(&image, height, width, &boxes, 0, None, 1);
        assert_eq!(result, image);
    }

    #[test]
    fn test_draw_boxes_clipping() {
        let height = 10;
        let width = 10;
        let image = vec![0u8; 3 * height * width];
        // Box extends beyond image boundaries
        let boxes = vec![-2.0, -2.0, 12.0, 12.0];
        let result = draw_boxes_slice(&image, height, width, &boxes, 1, None, 1);
        // Should not panic, and the result should have the same length
        assert_eq!(result.len(), image.len());
    }
}
