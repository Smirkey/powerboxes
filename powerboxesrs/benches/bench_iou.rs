use codspeed_criterion_compat::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use powerboxesrs::giou::{giou_distance, parallel_giou_distance};
use powerboxesrs::iou::{iou_distance, parallel_iou_distance};

pub fn iou_distance_benchmark(c: &mut Criterion) {
    let mut boxes1 = Array2::<f64>::zeros((100, 4));
    for i in 0..100 {
        for j in 0..4 {
            if j < 2 {
                boxes1[[i, j]] = 0.0;
            } else {
                boxes1[[i, j]] = 10.0;
            }
        }
    }
    let boxes2 = boxes1.clone();

    c.bench_function("iou distance benchmark", |b| {
        b.iter(|| iou_distance(black_box(&boxes1), black_box(&boxes2)))
    });
}

pub fn parallel_iou_distance_benchmark(c: &mut Criterion) {
    let mut boxes1 = Array2::<f64>::zeros((100, 4));
    for i in 0..100 {
        for j in 0..4 {
            if j < 2 {
                boxes1[[i, j]] = 0.0;
            } else {
                boxes1[[i, j]] = 10.0;
            }
        }
    }
    let boxes2 = boxes1.clone();

    c.bench_function("parallel iou distance benchmark", |b| {
        b.iter(|| parallel_iou_distance(black_box(&boxes1), black_box(&boxes2)))
    });
}

pub fn giou_distance_benchmark(c: &mut Criterion) {
    let mut boxes1 = Array2::<f64>::zeros((100, 4));
    for i in 0..100 {
        for j in 0..4 {
            if j < 2 {
                boxes1[[i, j]] = 0.0;
            } else {
                boxes1[[i, j]] = 10.0;
            }
        }
    }
    let boxes2 = boxes1.clone();

    c.bench_function("giou distance benchmark", |b| {
        b.iter(|| giou_distance(black_box(&boxes1), black_box(&boxes2)))
    });
}

pub fn parallel_giou_distance_benchmark(c: &mut Criterion) {
    let mut boxes1 = Array2::<f64>::zeros((100, 4));
    for i in 0..100 {
        for j in 0..4 {
            if j < 2 {
                boxes1[[i, j]] = 0.0;
            } else {
                boxes1[[i, j]] = 10.0;
            }
        }
    }
    let boxes2 = boxes1.clone();

    c.bench_function("parallel giou distance benchmark", |b| {
        b.iter(|| parallel_giou_distance(black_box(&boxes1), black_box(&boxes2)))
    });
}

criterion_group!(
    benches,
    iou_distance_benchmark,
    parallel_iou_distance_benchmark,
    giou_distance_benchmark,
    parallel_giou_distance_benchmark
);
criterion_main!(benches);
