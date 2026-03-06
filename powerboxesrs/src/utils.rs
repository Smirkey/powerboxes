use num_traits::{Num, ToPrimitive};
use rstar::{RStarInsertionStrategy, RTreeNum, RTreeObject, RTreeParams, AABB};

pub const ONE: f64 = 1.0;
pub const ZERO: f64 = 0.0;

#[inline]
pub fn min<N>(a: N, b: N) -> N
where
    N: Num + PartialOrd,
{
    if a < b {
        a
    } else {
        b
    }
}

#[inline]
pub fn max<N>(a: N, b: N) -> N
where
    N: Num + PartialOrd,
{
    if a > b {
        a
    } else {
        b
    }
}

/// Get a row from a flat slice representing an N x 4 matrix.
#[inline]
pub fn row4<N>(data: &[N], i: usize) -> (N, N, N, N)
where
    N: Copy,
{
    let base = i * 4;
    (data[base], data[base + 1], data[base + 2], data[base + 3])
}

/// Get a row from a flat slice representing an N x 5 matrix.
#[inline]
pub fn row5<N>(data: &[N], i: usize) -> (N, N, N, N, N)
where
    N: Copy,
{
    let base = i * 5;
    (
        data[base],
        data[base + 1],
        data[base + 2],
        data[base + 3],
        data[base + 4],
    )
}

// Struct we use to represent a bbox object in rstar R-tree
pub struct Bbox<T> {
    pub index: usize,
    pub x1: T,
    pub y1: T,
    pub x2: T,
    pub y2: T,
}

// Implement RTreeObject for Bbox
impl<T> RTreeObject for Bbox<T>
where
    T: RTreeNum + ToPrimitive + Sync + Send,
{
    type Envelope = AABB<[T; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners([self.x1, self.y1], [self.x2, self.y2])
    }
}

impl<T> RTreeParams for Bbox<T>
where
    T: RTreeNum + ToPrimitive + Sync + Send,
{
    const MIN_SIZE: usize = 16;
    const MAX_SIZE: usize = 256;
    const REINSERTION_COUNT: usize = 5;
    type DefaultInsertionStrategy = RStarInsertionStrategy;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min() {
        assert_eq!(min(1, 2), 1);
        assert_eq!(min(2, 1), 1);
        assert_eq!(min(2, 2), 2);
        assert_eq!(min(1., 2.), 1.);
        assert_eq!(min(2., 1.), 1.);
        assert_eq!(min(2., 2.), 2.);
    }
    #[test]
    fn test_max() {
        assert_eq!(max(1, 2), 2);
        assert_eq!(max(2, 1), 2);
        assert_eq!(max(2, 2), 2);
        assert_eq!(max(1., 2.), 2.);
        assert_eq!(max(2., 1.), 2.);
        assert_eq!(max(2., 2.), 2.);
    }

    #[test]
    fn test_row4() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(row4(&data, 0), (1.0, 2.0, 3.0, 4.0));
        assert_eq!(row4(&data, 1), (5.0, 6.0, 7.0, 8.0));
    }
}
