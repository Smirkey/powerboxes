use std::f64::consts::PI;
use std::ops::{Add, Sub};
use wide::f64x4;

/// A simple 2D point structure with x and y coordinates.
#[derive(Clone, Copy)]
pub struct Point {
    /// The x-coordinate of the point.
    pub x: f64,
    /// The y-coordinate of the point.
    pub y: f64,
}

impl Add for Point {
    type Output = Point;

    /// Adds two points component-wise and returns a new point.
    fn add(self, other: Point) -> Point {
        Point::new(self.x + other.x, self.y + other.y)
    }
}

impl Sub for Point {
    type Output = Point;

    /// Subtracts two points component-wise and returns a new point.
    fn sub(self, other: Point) -> Point {
        Point::new(self.x - other.x, self.y - other.y)
    }
}

impl Point {
    /// Creates a new point with the given x and y coordinates.
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    /// Computes the cross product of two points.
    ///
    /// # Returns
    /// The cross product, which is a scalar value.
    pub fn cross(self, other: Point) -> f64 {
        self.x * other.y - self.y * other.x
    }
}

/// Converts a bounding box specified by its center (cx, cy), width (w), height (h),
/// and rotation angle (r in degrees) to the coordinates of its four corner points.
/// The rotation is applied around the center of the bounding box.
///
/// # Arguments
///
/// * `cx` - X-coordinate of the center of the bounding box.
/// * `cy` - Y-coordinate of the center of the bounding box.
/// * `w` - Width of the bounding box.
/// * `h` - Height of the bounding box.
/// * `r` - Rotation angle in degrees.
///
/// # Returns
///
/// A tuple containing four `Point` instances representing the coordinates of the
/// four corners of the rotated bounding box in the following order: top-left, top-right,
/// bottom-right, bottom-left.
///
pub fn cxcywha_to_points(cx: f64, cy: f64, w: f64, h: f64, a: f64) -> (Point, Point, Point, Point) {
    let radians = PI * a / 180.0;
    let dx = w / 2.0;
    let dy = h / 2.0;
    let sin_radians = radians.sin();
    let cos_radians = radians.cos();
    let dxcos = dx * cos_radians;
    let dxsin = dx * sin_radians;
    let dycos = dy * cos_radians;
    let dysin = dy * sin_radians;

    let (p1, p2, p3, p4) = (
        Point::new(cx, cy) + Point::new(-dxcos - -dysin, -dxsin + -dycos),
        Point::new(cx, cy) + Point::new(dxcos - -dysin, dxsin + -dycos),
        Point::new(cx, cy) + Point::new(dxcos - dysin, dxsin + dycos),
        Point::new(cx, cy) + Point::new(-dxcos - dysin, -dxsin + dycos),
    );
    return (p1, p2, p3, p4);
}

#[derive(Clone, Copy)]
pub struct Rect {
    pub p1: Point,
    pub p2: Point,
    pub p3: Point,
    pub p4: Point,
}

impl Rect {
    pub fn new(cx: f64, cy: f64, w: f64, h: f64, a: f64) -> Self {
        let (p1, p2, p3, p4) = cxcywha_to_points(cx, cy, w, h, a);
        Rect { p1, p2, p3, p4 }
    }
    pub fn points(&self) -> [Point; 4] {
        [self.p1, self.p2, self.p3, self.p4]
    }
}

/// Represents a 2D line in the form Ax + By + C = 0, where A, B, and C are coefficients.
///
/// The `Line` struct provides methods for creating a new line from two points, evaluating the line
/// equation at a given point, and finding the intersection point with another line.
///
/// # Examples
///
/// ```
/// use powerboxesrs::rotation::{Line, Point};
///
/// let p1 = Point::new(0.0, 0.0);
/// let p2 = Point::new(1.0, 1.0);
/// let line = Line::new(p1, p2);
///
/// let point_on_line = Point::new(0.5, 0.5);
/// let value_at_point = line.call(point_on_line);
///
/// let another_line = Line::new(Point::new(0.0, 1.0), Point::new(1.0, 0.0));
/// let intersection_point = line.intersection(&another_line);
/// ```
///
#[derive(Debug)]
pub struct Line {
    /// Coefficient A in the line equation.
    pub a: f64,

    /// Coefficient B in the line equation.
    pub b: f64,

    /// Coefficient C in the line equation.
    pub c: f64,
}

impl Line {
    /// Creates a new line from two points.
    ///
    /// The line equation is determined as A(x - x₁) + B(y - y₁) = 0, where (x₁, y₁) and (x₂, y₂)
    /// are the coordinates of the two points.
    ///
    /// # Arguments
    ///
    /// * `p1` - The first point.
    /// * `p2` - The second point.
    ///
    /// # Returns
    ///
    /// Returns a new `Line` struct.
    pub fn new(p1: Point, p2: Point) -> Self {
        let a = p2.y - p1.y;
        let b = p1.x - p2.x;
        let c = p2.cross(p1);
        Line { a, b, c }
    }

    /// Evaluates the line equation at a given point.
    ///
    /// # Arguments
    ///
    /// * `p` - The point at which to evaluate the line equation.
    ///
    /// # Returns
    ///
    /// Returns the value of the line equation at the specified point.
    pub fn call(&self, p: Point) -> f64 {
        self.a * p.x + self.b * p.y + self.c
    }

    /// Finds the intersection point with another line.
    ///
    /// # Arguments
    ///
    /// * `other` - The other line to find the intersection with.
    ///
    /// # Returns
    ///
    /// Returns a `Point` representing the intersection point of the two lines.
    pub fn intersection(&self, other: &Line) -> Point {
        let w = self.a * other.b - self.b * other.a;
        Point::new(
            (self.b * other.c - self.c * other.b) / w,
            (self.c * other.a - self.a * other.c) / w,
        )
    }
}

/// Max polygon vertices after clipping a 4-gon by 4 edges (each edge adds at most 1 vertex).
const MAX_POLY_VERTS: usize = 8;

/// SoA polygon representation on the stack.
struct SoaPoly {
    xs: [f64; MAX_POLY_VERTS],
    ys: [f64; MAX_POLY_VERTS],
    len: usize,
}

impl SoaPoly {
    #[inline]
    fn from_rect(rect: &Rect) -> Self {
        let pts = rect.points();
        let mut xs = [0.0; MAX_POLY_VERTS];
        let mut ys = [0.0; MAX_POLY_VERTS];
        for (i, p) in pts.iter().enumerate() {
            xs[i] = p.x;
            ys[i] = p.y;
        }
        SoaPoly { xs, ys, len: 4 }
    }

    /// Batch-evaluate line equation for all vertices using SIMD.
    /// Returns values in a fixed-size buffer.
    #[inline]
    fn line_values(&self, line: &Line) -> ([f64; MAX_POLY_VERTS], usize) {
        let mut values = [0.0f64; MAX_POLY_VERTS];
        let a_vec = f64x4::splat(line.a);
        let b_vec = f64x4::splat(line.b);
        let c_vec = f64x4::splat(line.c);

        // Process first 4 vertices with SIMD
        if self.len >= 4 {
            let xs = f64x4::from([self.xs[0], self.xs[1], self.xs[2], self.xs[3]]);
            let ys = f64x4::from([self.ys[0], self.ys[1], self.ys[2], self.ys[3]]);
            let result = a_vec * xs + b_vec * ys + c_vec;
            let arr: [f64; 4] = result.into();
            values[0] = arr[0];
            values[1] = arr[1];
            values[2] = arr[2];
            values[3] = arr[3];
        } else {
            for i in 0..self.len {
                values[i] = line.a * self.xs[i] + line.b * self.ys[i] + line.c;
            }
            return (values, self.len);
        }

        // Process remaining vertices (5-8) with SIMD if we have them
        if self.len > 4 {
            let remaining = self.len - 4;
            let mut rx = [0.0f64; 4];
            let mut ry = [0.0f64; 4];
            for i in 0..remaining {
                rx[i] = self.xs[4 + i];
                ry[i] = self.ys[4 + i];
            }
            let xs = f64x4::from(rx);
            let ys = f64x4::from(ry);
            let result = a_vec * xs + b_vec * ys + c_vec;
            let arr: [f64; 4] = result.into();
            for i in 0..remaining {
                values[4 + i] = arr[i];
            }
        }

        (values, self.len)
    }

    /// Compute polygon area using the shoelace formula with SIMD.
    #[inline]
    fn area(&self) -> f64 {
        if self.len <= 2 {
            return 0.0;
        }
        let n = self.len;

        // Build shifted arrays: xs_next[i] = xs[(i+1) % n], ys_next[i] = ys[(i+1) % n]
        let mut xs_next = [0.0f64; MAX_POLY_VERTS];
        let mut ys_next = [0.0f64; MAX_POLY_VERTS];
        for i in 0..n - 1 {
            xs_next[i] = self.xs[i + 1];
            ys_next[i] = self.ys[i + 1];
        }
        xs_next[n - 1] = self.xs[0];
        ys_next[n - 1] = self.ys[0];

        // SIMD shoelace: sum of (x_i * y_{i+1} - y_i * x_{i+1})
        let mut sum = 0.0f64;

        // Process 4 at a time
        if n >= 4 {
            let x = f64x4::from([self.xs[0], self.xs[1], self.xs[2], self.xs[3]]);
            let y = f64x4::from([self.ys[0], self.ys[1], self.ys[2], self.ys[3]]);
            let xn = f64x4::from([xs_next[0], xs_next[1], xs_next[2], xs_next[3]]);
            let yn = f64x4::from([ys_next[0], ys_next[1], ys_next[2], ys_next[3]]);
            let cross = x * yn - y * xn;
            let arr: [f64; 4] = cross.into();
            sum += arr[0] + arr[1] + arr[2] + arr[3];
        }

        // Process remaining (indices 4..n) with SIMD if possible
        if n > 4 {
            let remaining = n - 4;
            let mut rx = [0.0f64; 4];
            let mut ry = [0.0f64; 4];
            let mut rxn = [0.0f64; 4];
            let mut ryn = [0.0f64; 4];
            for i in 0..remaining {
                rx[i] = self.xs[4 + i];
                ry[i] = self.ys[4 + i];
                rxn[i] = xs_next[4 + i];
                ryn[i] = ys_next[4 + i];
            }
            let x = f64x4::from(rx);
            let y = f64x4::from(ry);
            let xn = f64x4::from(rxn);
            let yn = f64x4::from(ryn);
            let cross = x * yn - y * xn;
            let arr: [f64; 4] = cross.into();
            for i in 0..remaining {
                sum += arr[i];
            }
        } else if n < 4 {
            // Small polygon (3 vertices) — scalar fallback
            for i in 0..n {
                sum += self.xs[i] * ys_next[i] - self.ys[i] * xs_next[i];
            }
            return 0.5 * sum;
        }

        0.5 * sum
    }
}

/// Calculates the area of intersection between two rectangles represented by `Rect` structs.
///
/// The function takes two rectangles, `rect1` and `rect2`, and computes the area of their intersection.
/// The rectangles are assumed to be represented as structures containing information about their points.
///
/// # Arguments
///
/// * `rect1` - The first rectangle.
/// * `rect2` - The second rectangle.
///
/// # Returns
///
/// Returns a `f64` representing the area of intersection between the two rectangles.
///
/// # Notes
///
/// This implementation utilizes the Sutherland-Hodgman algorithm for computing the intersection area.
/// All polygon buffers are stack-allocated (zero heap allocations). Line evaluations use SIMD via
/// the `wide` crate for f64x4 vectorization.
///
pub fn intersection_area(rect1: &Rect, rect2: &Rect) -> f64 {
    let mut poly = SoaPoly::from_rect(rect1);

    let r2_pts = rect2.points();

    // Clip against each edge of rect2
    for edge_idx in 0..4 {
        if poly.len <= 2 {
            return 0.0;
        }

        let next_idx = (edge_idx + 1) & 3; // mod 4
        let line = Line::new(r2_pts[edge_idx], r2_pts[next_idx]);

        // Batch evaluate line for all polygon vertices (SIMD)
        let (line_vals, n) = poly.line_values(&line);

        let mut new_xs = [0.0f64; MAX_POLY_VERTS];
        let mut new_ys = [0.0f64; MAX_POLY_VERTS];
        let mut new_len: usize = 0;

        // Sutherland-Hodgman clipping (sequential — branch-heavy)
        for i in 0..n {
            let next_i = if i + 1 < n { i + 1 } else { 0 };
            let s_val = line_vals[i];
            let t_val = line_vals[next_i];

            if s_val <= 0.0 {
                new_xs[new_len] = poly.xs[i];
                new_ys[new_len] = poly.ys[i];
                new_len += 1;
            }
            if s_val * t_val < 0.0 {
                let s_pt = Point::new(poly.xs[i], poly.ys[i]);
                let t_pt = Point::new(poly.xs[next_i], poly.ys[next_i]);
                let intersection_pt = line.intersection(&Line::new(s_pt, t_pt));
                new_xs[new_len] = intersection_pt.x;
                new_ys[new_len] = intersection_pt.y;
                new_len += 1;
            }
        }

        poly.xs = new_xs;
        poly.ys = new_ys;
        poly.len = new_len;
    }

    poly.area()
}

pub fn minimal_bounding_rect(points: &[Point]) -> (f64, f64, f64, f64) {
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);

    for point in points {
        min_x = min_x.min(point.x);
        min_y = min_y.min(point.y);
        max_x = max_x.max(point.x);
        max_y = max_y.max(point.y);
    }
    (min_x, min_y, max_x, max_y)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_rotated_intersection_normal_case() {
        let r1 = Rect::new(10., 15., 15., 10., 30.);
        let r2 = Rect::new(15., 15., 20., 10., 0.);
        let intersection = intersection_area(&r1, &r2);
        assert_eq!(intersection, 110.17763185469022);
    }

    #[test]
    fn test_rotated_intersection_zero_intersection() {
        let r1 = Rect::new(10., 15., 15., 10., 30.);
        let r2 = Rect::new(150., 150., 20., 10., 0.);
        let intersection = intersection_area(&r1, &r2);
        assert_eq!(intersection, 0.0);
    }

    #[test]
    fn test_rotated_intersection_max_intersection() {
        let r1 = Rect::new(150., 150., 20., 10., 0.);
        let r2 = Rect::new(150., 150., 20., 10., 0.);
        let intersection = intersection_area(&r1, &r2);
        assert_eq!(intersection, 200.0);
    }
}
