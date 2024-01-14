use std::f64::consts::PI;
use std::ops::{Add, Sub};

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
    pub fn points(&self) -> Vec<Point> {
        vec![self.p1, self.p2, self.p3, self.p4]
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
/// This implementation utilizes the Separating Axis Theorem (SAT) for computing the intersection area.
///
pub fn intersection_area(rect1: Rect, rect2: Rect) -> f64 {
    let mut intersection = rect1.points();

    for (p, q) in rect2
        .points()
        .into_iter()
        .zip(rect2.points().into_iter().cycle().skip(1))
    {
        if intersection.len() <= 2 {
            break;
        }

        let line = Line::new(p, q);

        let mut new_intersection = Vec::new();
        let line_values: Vec<f64> = intersection.iter().map(|t| line.call(*t)).collect();
        let truncated_intersection: Vec<&Point> =
            intersection[1..].iter().chain(&intersection[..1]).collect();
        let truncated_line_values: Vec<&f64> =
            line_values[1..].iter().chain(&line_values[..1]).collect();
        for (((s, t), s_value), t_value) in intersection
            .iter()
            .zip(truncated_intersection)
            .zip(&line_values)
            .zip(truncated_line_values)
        {
            if s_value <= &0.0 {
                new_intersection.push(*s)
            }
            if s_value * t_value < 0.0 {
                let intersection_point = line.intersection(&Line::new(*s, *t));
                new_intersection.push(intersection_point);
            }
        }
        intersection = new_intersection;
    }

    if intersection.len() <= 2 {
        return 0.0;
    }
    let truncated_intersection: Vec<&Point> =
        intersection[1..].iter().chain(&intersection[..1]).collect();
    return 0.5
        * intersection
            .iter()
            .zip(truncated_intersection.iter())
            .map(|(&p, &q)| p.x * q.y - p.y * q.x)
            .sum::<f64>();
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_rotated_intersection_normal_case() {
        let r1 = Rect::new(10., 15., 15., 10., 30.);
        let r2 = Rect::new(15., 15., 20., 10., 0.);
        let intersection = intersection_area(r1, r2);
        assert_eq!(intersection, 110.17763185469022);
    }

    #[test]
    fn test_rotated_intersection_zero_intersection() {
        let r1 = Rect::new(10., 15., 15., 10., 30.);
        let r2 = Rect::new(150., 150., 20., 10., 0.);
        let intersection = intersection_area(r1, r2);
        assert_eq!(intersection, 0.0);
    }

    #[test]
    fn test_rotated_intersection_max_intersection() {
        let r1 = Rect::new(150., 150., 20., 10., 0.);
        let r2 = Rect::new(150., 150., 20., 10., 0.);
        let intersection = intersection_area(r1, r2);
        assert_eq!(intersection, 200.0);
    }
}
