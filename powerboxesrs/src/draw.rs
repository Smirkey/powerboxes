use crate::rotation::{cxcywha_to_points, Point};

const DEFAULT_COLORS: [(u8, u8, u8); 10] = [
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

#[derive(Clone, Copy, Debug)]
pub struct DrawOptions {
    pub thickness: usize,
    pub filled: bool,
    pub opacity: f64,
}

impl DrawOptions {
    #[inline]
    fn normalized(self) -> Self {
        Self {
            thickness: normalize_thickness(self.thickness),
            filled: self.filled,
            opacity: self.opacity.clamp(0.0, 1.0),
        }
    }
}

impl Default for DrawOptions {
    fn default() -> Self {
        Self {
            thickness: 2,
            filled: false,
            opacity: 1.0,
        }
    }
}

/// Draw axis-aligned bounding boxes on a CHW image tensor.
pub fn draw_boxes_slice(
    image: &[u8],
    height: usize,
    width: usize,
    boxes: &[f64],
    num_boxes: usize,
    colors: Option<&[u8]>,
    options: DrawOptions,
) -> Vec<u8> {
    let mut output = image.to_vec();
    let options = options.normalized();

    for b in 0..num_boxes {
        let base = b * 4;
        let x1 = boxes[base].round().max(0.0) as i32;
        let y1 = boxes[base + 1].round().max(0.0) as i32;
        let x2 = boxes[base + 2].round().min(width as f64 - 1.0).max(0.0) as i32;
        let y2 = boxes[base + 3].round().min(height as f64 - 1.0).max(0.0) as i32;

        if x1 > x2 || y1 > y2 || x1 >= width as i32 || y1 >= height as i32 {
            continue;
        }

        let color = box_color(colors, b);
        if options.filled {
            fill_rect(
                &mut output,
                height,
                width,
                x1,
                y1,
                x2,
                y2,
                color,
                options.opacity,
            );
        }
        draw_rect_outline(
            &mut output,
            height,
            width,
            x1,
            y1,
            x2,
            y2,
            color,
            options.thickness,
            options.opacity,
        );
    }

    output
}

/// Draw rotated bounding boxes on a CHW image tensor.
pub fn draw_rotated_boxes_slice(
    image: &[u8],
    height: usize,
    width: usize,
    boxes: &[f64],
    num_boxes: usize,
    colors: Option<&[u8]>,
    options: DrawOptions,
) -> Vec<u8> {
    let mut output = image.to_vec();
    let options = options.normalized();

    for b in 0..num_boxes {
        let base = b * 5;
        let points = {
            let (p1, p2, p3, p4) = cxcywha_to_points(
                boxes[base],
                boxes[base + 1],
                boxes[base + 2],
                boxes[base + 3],
                boxes[base + 4],
            );
            [p1, p2, p3, p4]
        };
        let color = box_color(colors, b);

        if options.filled {
            fill_convex_quad(&mut output, height, width, &points, color, options.opacity);
        }

        for edge in 0..4 {
            let start = points[edge];
            let end = points[(edge + 1) % 4];
            draw_line(
                &mut output,
                height,
                width,
                start,
                end,
                color,
                options.thickness,
                options.opacity,
            );
        }
    }

    output
}

#[inline]
fn normalize_thickness(thickness: usize) -> usize {
    if thickness == 0 {
        1
    } else {
        thickness
    }
}

#[inline]
fn box_color(colors: Option<&[u8]>, index: usize) -> (u8, u8, u8) {
    if let Some(c) = colors {
        let base = index * 3;
        (c[base], c[base + 1], c[base + 2])
    } else {
        DEFAULT_COLORS[index % DEFAULT_COLORS.len()]
    }
}

#[inline]
fn pixel_index(width: usize, y: usize, x: usize) -> usize {
    y * width + x
}

#[inline]
fn blend_channel(dst: u8, src: u8, opacity: f64) -> u8 {
    if opacity >= 1.0 {
        src
    } else {
        (opacity * f64::from(src) + (1.0 - opacity) * f64::from(dst)).round() as u8
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn set_pixel_chw(
    image: &mut [u8],
    height: usize,
    width: usize,
    y: usize,
    x: usize,
    r: u8,
    g: u8,
    b: u8,
    opacity: f64,
) {
    let hw = height * width;
    let idx = pixel_index(width, y, x);
    image[idx] = blend_channel(image[idx], r, opacity);
    image[hw + idx] = blend_channel(image[hw + idx], g, opacity);
    image[2 * hw + idx] = blend_channel(image[2 * hw + idx], b, opacity);
}

#[allow(clippy::too_many_arguments)]
fn fill_rect(
    image: &mut [u8],
    height: usize,
    width: usize,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    color: (u8, u8, u8),
    opacity: f64,
) {
    for y in y1.max(0)..=y2.min(height as i32 - 1) {
        for x in x1.max(0)..=x2.min(width as i32 - 1) {
            set_pixel_chw(
                image, height, width, y as usize, x as usize, color.0, color.1, color.2, opacity,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_rect_outline(
    image: &mut [u8],
    height: usize,
    width: usize,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    color: (u8, u8, u8),
    thickness: usize,
    opacity: f64,
) {
    let half_t = (thickness as i32) / 2;

    for x in x1..=x2 {
        for dy in 0..thickness as i32 {
            let yt = y1 + dy - half_t;
            let yb = y2 + dy - half_t;
            draw_if_in_bounds(image, height, width, x, yt, color, opacity);
            draw_if_in_bounds(image, height, width, x, yb, color, opacity);
        }
    }

    for y in y1..=y2 {
        for dx in 0..thickness as i32 {
            let xl = x1 + dx - half_t;
            let xr = x2 + dx - half_t;
            draw_if_in_bounds(image, height, width, xl, y, color, opacity);
            draw_if_in_bounds(image, height, width, xr, y, color, opacity);
        }
    }
}

#[inline]
fn draw_if_in_bounds(
    image: &mut [u8],
    height: usize,
    width: usize,
    x: i32,
    y: i32,
    color: (u8, u8, u8),
    opacity: f64,
) {
    if x >= 0 && y >= 0 && x < width as i32 && y < height as i32 {
        set_pixel_chw(
            image, height, width, y as usize, x as usize, color.0, color.1, color.2, opacity,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_line(
    image: &mut [u8],
    height: usize,
    width: usize,
    start: Point,
    end: Point,
    color: (u8, u8, u8),
    thickness: usize,
    opacity: f64,
) {
    let mut x0 = start.x.round() as i32;
    let mut y0 = start.y.round() as i32;
    let x1 = end.x.round() as i32;
    let y1 = end.y.round() as i32;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let half_t = (thickness as i32) / 2;

    loop {
        for oy in -half_t..=half_t {
            for ox in -half_t..=half_t {
                draw_if_in_bounds(image, height, width, x0 + ox, y0 + oy, color, opacity);
            }
        }

        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn fill_convex_quad(
    image: &mut [u8],
    height: usize,
    width: usize,
    points: &[Point; 4],
    color: (u8, u8, u8),
    opacity: f64,
) {
    let min_x = points
        .iter()
        .map(|p| p.x.floor() as i32)
        .min()
        .unwrap_or(0)
        .max(0);
    let max_x = points
        .iter()
        .map(|p| p.x.ceil() as i32)
        .max()
        .unwrap_or(-1)
        .min(width as i32 - 1);
    let min_y = points
        .iter()
        .map(|p| p.y.floor() as i32)
        .min()
        .unwrap_or(0)
        .max(0);
    let max_y = points
        .iter()
        .map(|p| p.y.ceil() as i32)
        .max()
        .unwrap_or(-1)
        .min(height as i32 - 1);

    if min_x > max_x || min_y > max_y {
        return;
    }

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if point_in_convex_quad(x as f64 + 0.5, y as f64 + 0.5, points) {
                set_pixel_chw(
                    image, height, width, y as usize, x as usize, color.0, color.1, color.2,
                    opacity,
                );
            }
        }
    }
}

fn point_in_convex_quad(x: f64, y: f64, points: &[Point; 4]) -> bool {
    let mut sign = 0.0;
    for edge in 0..4 {
        let p1 = points[edge];
        let p2 = points[(edge + 1) % 4];
        let cross = (p2.x - p1.x) * (y - p1.y) - (p2.y - p1.y) * (x - p1.x);
        if cross.abs() < f64::EPSILON {
            continue;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if cross.signum() != sign {
            return false;
        }
    }
    true
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
        let result = draw_boxes_slice(
            &image,
            height,
            width,
            &boxes,
            1,
            None,
            DrawOptions {
                thickness: 1,
                ..DrawOptions::default()
            },
        );

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
        let result = draw_boxes_slice(
            &image,
            height,
            width,
            &boxes,
            1,
            Some(&colors),
            DrawOptions {
                thickness: 1,
                ..DrawOptions::default()
            },
        );

        let hw = height * width;
        // Check top-left corner of box (1, 1) has custom color
        assert_eq!(result[1 * width + 1], 0); // R
        assert_eq!(result[hw + 1 * width + 1], 255); // G
        assert_eq!(result[2 * hw + 1 * width + 1], 128); // B
    }

    #[test]
    fn test_draw_boxes_empty() {
        let height = 10;
        let width = 10;
        let image = vec![0u8; 3 * height * width];
        let boxes: Vec<f64> = vec![];
        let result = draw_boxes_slice(
            &image,
            height,
            width,
            &boxes,
            0,
            None,
            DrawOptions {
                thickness: 1,
                ..DrawOptions::default()
            },
        );
        assert_eq!(result, image);
    }

    #[test]
    fn test_draw_boxes_clipping() {
        let height = 10;
        let width = 10;
        let image = vec![0u8; 3 * height * width];
        // Box extends beyond image boundaries
        let boxes = vec![-2.0, -2.0, 12.0, 12.0];
        let result = draw_boxes_slice(
            &image,
            height,
            width,
            &boxes,
            1,
            None,
            DrawOptions {
                thickness: 1,
                ..DrawOptions::default()
            },
        );
        // Should not panic, and the result should have the same length
        assert_eq!(result.len(), image.len());
    }

    #[test]
    fn test_draw_boxes_filled_with_opacity() {
        let height = 8;
        let width = 8;
        let image = vec![100u8; 3 * height * width];
        let boxes = vec![1.0, 1.0, 5.0, 5.0];
        let result = draw_boxes_slice(
            &image,
            height,
            width,
            &boxes,
            1,
            None,
            DrawOptions {
                thickness: 1,
                filled: true,
                opacity: 0.5,
            },
        );

        assert_eq!(result[3 * width + 3], 178);
        assert_eq!(result[height * width + 3 * width + 3], 50);
        assert_eq!(result[2 * height * width + 3 * width + 3], 50);
    }

    #[test]
    fn test_draw_rotated_boxes_slice() {
        let height = 20;
        let width = 20;
        let image = vec![0u8; 3 * height * width];
        let boxes = vec![10.0, 10.0, 8.0, 4.0, 30.0];
        let result = draw_rotated_boxes_slice(
            &image,
            height,
            width,
            &boxes,
            1,
            None,
            DrawOptions {
                thickness: 1,
                ..DrawOptions::default()
            },
        );

        assert!(result.iter().any(|&v| v != 0));
    }
}
