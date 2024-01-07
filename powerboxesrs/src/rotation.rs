use std::f64::consts::PI;
use std::ops::{Add, Sub};
use std::vec::IntoIter;

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

impl Add for Point {
    type Output = Point;

    fn add(self, other: Point) -> Point {
        Point::new(self.x + other.x, self.y + other.y)
    }
}

impl Sub for Point {
    type Output = Point;

    fn sub(self, other: Point) -> Point {
        Point::new(self.x - other.x, self.y - other.y)
    }
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
    fn cross(self, other: Point) -> f64 {
        return self.x * other.y - self.y * other.x;
    }
}

#[derive(Clone, Copy)]
struct Rect {
    p1: Point,
    p2: Point,
    p3: Point,
    p4: Point,
}

impl Rect {
    fn new(cx: f64, cy: f64, w: f64, h: f64, r: f64) -> Self {
        let angle = PI * r / 180.0;
        let dx = w / 2.0;
        let dy = h / 2.0;
        let dxcos = dx * angle.cos();
        let dxsin = dx * angle.sin();
        let dycos = dy * angle.cos();
        let dysin = dy * angle.sin();

        let (p1, p2, p3, p4) = (
            Point::new(cx, cy) + Point::new(-dxcos - -dysin, -dxsin + -dycos),
            Point::new(cx, cy) + Point::new(dxcos - -dysin, dxsin + -dycos),
            Point::new(cx, cy) + Point::new(dxcos - dysin, dxsin + dycos),
            Point::new(cx, cy) + Point::new(-dxcos - dysin, -dxsin + dycos),
        );
        Rect { p1, p2, p3, p4 }
    }

    fn area(&self) -> f64 {
        let side1 = (self.p1.x - self.p2.x).abs();
        let side2 = (self.p1.y - self.p4.y).abs();
        side1 * side2
    }
    fn points(&self) -> Vec<Point> {
        vec![self.p1, self.p2, self.p3, self.p4]
    }
}

impl IntoIterator for Rect {
    type Item = Point;
    type IntoIter = IntoIter<Point>;

    fn into_iter(self) -> Self::IntoIter {
        vec![self.p1, self.p2, self.p3, self.p4].into_iter()
    }
}

struct Line {
    a: f64,
    b: f64,
    c: f64,
}

impl Line {
    fn new(p1: Point, p2: Point) -> Self {
        let a = p2.y - p1.y;
        let b = p1.x - p2.x;
        let c = p2.cross(p1);
        Line { a, b, c }
    }

    fn call(&self, p: Point) -> f64 {
        self.a * p.x + self.b * p.y + self.c
    }

    fn intersection(&self, other: &Line) -> Point {
        let w = self.a * other.b - self.b * other.a;
        Point::new(
            (self.b * other.c - self.c * other.b) / w,
            (self.c * other.a - self.a * other.c) / w,
        )
    }
}

fn intersection_area(r1: (f64, f64, f64, f64, f64), r2: (f64, f64, f64, f64, f64)) -> f64 {
    let rect1 = Rect::new(r1.0, r1.1, r1.2, r1.3, r1.4);
    let rect2 = Rect::new(r2.0, r2.1, r2.2, r2.3, r2.4);

    let mut intersection = rect1.points();

    for (p, q) in rect2.into_iter().zip(rect2.into_iter().cycle().skip(1)) {
        if intersection.len() <= 2 {
            break;
        }

        let line = Line::new(p, q);

        let mut new_intersection = Vec::new();
        let line_values: Vec<f64> = intersection.iter().map(|t| line.call(*t)).collect();
        let truncated_intersection: Vec<Point> = intersection[1..]
            .iter()
            .chain(&intersection[..1])
            .cloned()
            .collect();
        let truncated_line_values: Vec<f64> = line_values[1..]
            .iter()
            .chain(&line_values[..1])
            .cloned()
            .collect();
        for (((s, t), s_value), t_value) in intersection
            .iter()
            .zip(truncated_intersection)
            .zip(line_values)
            .zip(truncated_line_values)
        {
            if s_value <= 0.0 {
                new_intersection.push(*s)
            }
            if s_value * t_value < 0.0 {
                let intersection_point = line.intersection(&Line::new(*s, t));
                new_intersection.push(intersection_point);
            }
        }
        intersection = new_intersection;
    }

    if intersection.len() <= 2 {
        return 0.0;
    }
    let truncated_intersection: Vec<Point> = intersection[1..]
        .iter()
        .chain(&intersection[..1])
        .cloned()
        .collect();
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
        let r1 = (10., 15., 15., 10., 30.);
        let r2 = (15., 15., 20., 10., 0.);
        let intersection = intersection_area(r1, r2);
        assert_eq!(intersection, 110.17763185469022);
    }
}
