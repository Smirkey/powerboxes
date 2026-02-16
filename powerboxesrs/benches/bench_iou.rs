use codspeed_criterion_compat::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use powerboxesrs::giou::{giou_distance, parallel_giou_distance, rotated_giou_distance, parallel_rotated_giou_distance};
use powerboxesrs::iou::{iou_distance, parallel_iou_distance, rotated_iou_distance, parallel_rotated_iou_distance};
use powerboxesrs::tiou::{rotated_tiou_distance, parallel_rotated_tiou_distance};

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

/// Generate 100x5 rotated boxes with varied angles for benchmarking
fn make_rotated_boxes(n: usize) -> Array2<f64> {
    let mut boxes = Array2::<f64>::zeros((n, 5));
    for i in 0..n {
        let offset = (i as f64) * 0.5;
        boxes[[i, 0]] = 10.0 + offset; // cx
        boxes[[i, 1]] = 10.0 + offset; // cy
        boxes[[i, 2]] = 5.0;           // w
        boxes[[i, 3]] = 3.0;           // h
        boxes[[i, 4]] = (i as f64 * 17.0) % 360.0; // angle (varied)
    }
    boxes
}

pub fn rotated_iou_distance_benchmark(c: &mut Criterion) {
    let boxes1 = make_rotated_boxes(100);
    let boxes2 = make_rotated_boxes(100);

    c.bench_function("rotated iou distance benchmark", |b| {
        b.iter(|| rotated_iou_distance(black_box(&boxes1), black_box(&boxes2)))
    });
}

pub fn parallel_rotated_iou_distance_benchmark(c: &mut Criterion) {
    let boxes1 = make_rotated_boxes(100);
    let boxes2 = make_rotated_boxes(100);

    c.bench_function("parallel rotated iou distance benchmark", |b| {
        b.iter(|| parallel_rotated_iou_distance(black_box(&boxes1), black_box(&boxes2)))
    });
}

pub fn rotated_giou_distance_benchmark(c: &mut Criterion) {
    let boxes1 = make_rotated_boxes(100);
    let boxes2 = make_rotated_boxes(100);

    c.bench_function("rotated giou distance benchmark", |b| {
        b.iter(|| rotated_giou_distance(black_box(&boxes1), black_box(&boxes2)))
    });
}

pub fn parallel_rotated_giou_distance_benchmark(c: &mut Criterion) {
    let boxes1 = make_rotated_boxes(100);
    let boxes2 = make_rotated_boxes(100);

    c.bench_function("parallel rotated giou distance benchmark", |b| {
        b.iter(|| parallel_rotated_giou_distance(black_box(&boxes1), black_box(&boxes2)))
    });
}

pub fn rotated_tiou_distance_benchmark(c: &mut Criterion) {
    let boxes1 = make_rotated_boxes(100);
    let boxes2 = make_rotated_boxes(100);

    c.bench_function("rotated tiou distance benchmark", |b| {
        b.iter(|| rotated_tiou_distance(black_box(&boxes1), black_box(&boxes2)))
    });
}

pub fn parallel_rotated_tiou_distance_benchmark(c: &mut Criterion) {
    let boxes1 = make_rotated_boxes(100);
    let boxes2 = make_rotated_boxes(100);

    c.bench_function("parallel rotated tiou distance benchmark", |b| {
        b.iter(|| parallel_rotated_tiou_distance(black_box(&boxes1), black_box(&boxes2)))
    });
}

criterion_group!(
    benches,
    iou_distance_benchmark,
    parallel_iou_distance_benchmark,
    giou_distance_benchmark,
    parallel_giou_distance_benchmark,
    rotated_iou_distance_benchmark,
    parallel_rotated_iou_distance_benchmark,
    rotated_giou_distance_benchmark,
    parallel_rotated_giou_distance_benchmark,
    rotated_tiou_distance_benchmark,
    parallel_rotated_tiou_distance_benchmark
);
criterion_main!(benches);
