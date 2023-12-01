use num_traits::Num;

pub const EPS: f64 = 1e-16;
pub const ONE: f64 = 1.0;
pub const ZERO: f64 = 0.0;

pub fn min<N>(a: N, b: N) -> N
where
    N: Num + PartialOrd,
{
    if a < b {
        return a;
    } else {
        return b;
    }
}

pub fn max<N>(a: N, b: N) -> N
where
    N: Num + PartialOrd,
{
    if a > b {
        return a;
    } else {
        return b;
    }
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
}
