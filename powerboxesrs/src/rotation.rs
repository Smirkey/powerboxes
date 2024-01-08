use std::f64::consts::PI;
use std::ops::{Add, Sub};

#[derive(Clone, Copy)]
pub struct Point {
    pub x: f64,
    pub y: f64,
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
}

pub fn cxcywha_to_points(cx: f64, cy: f64, w: f64, h: f64, r: f64) -> (Point, Point, Point, Point) {
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
    return (p1, p2, p3, p4);
}
