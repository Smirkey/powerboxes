@pytest.mark.benchmark(group="nms")
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_nms(benchmark, dtype, generate_boxes):
    boxes = generate_boxes
    boxes = boxes.astype(dtype)
    scores = np.ones(len(boxes)).astype(np.float64)
    benchmark(nms, boxes, scores, 0.5, 0.5)


@pytest.mark.benchmark(group="nms")
@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "int32", "int16"])
def test_rtree_nms(benchmark, dtype, generate_boxes):
    boxes = generate_boxes
    boxes = boxes.astype(dtype)
    scores = np.ones(len(boxes)).astype(np.float64)
    benchmark(rtree_nms, boxes, scores, 0.5, 0.5)


@pytest.mark.benchmark(group="nms_many_boxes")
@pytest.mark.parametrize("n_boxes", [1000, 5000, 10000])
def test_nms_many_boxes(benchmark, n_boxes, generate_boxes):
    boxes = generate_boxes
    scores = np.ones(len(boxes)).astype(np.float64)
    benchmark(nms, boxes, scores, 0.5, 0.5)


@pytest.mark.benchmark(group="nms_many_boxes")
@pytest.mark.parametrize("n_boxes", [1000, 5000, 10000])
def test_rtree_nms_many_boxes(benchmark, n_boxes, generate_boxes):
    boxes = generate_boxes
    scores = np.ones(len(boxes)).astype(np.float64)
    benchmark(rtree_nms, boxes, scores, 0.5, 0.5)
