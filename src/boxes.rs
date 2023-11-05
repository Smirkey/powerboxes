use ndarray::{Array1, Array2, Axis, Zip};

pub enum BoxFormat {
    XYXY,
    XYWH,
    CXCYWH,
}

pub fn box_areas(boxes: &Array2<f64>) -> Array1<f64> {
    let num_boxes = boxes.nrows();
    let mut areas = Array1::<f64>::zeros(num_boxes);

    Zip::indexed(&mut areas).for_each(|i, area| {
        let box1 = boxes.row(i);
        let x1 = box1[0];
        let y1 = box1[1];
        let x2 = box1[2];
        let y2 = box1[3];
        *area = (x2 - x1 + 1.) * (y2 - y1 + 1.);
    });

    return areas;
}

pub fn parallel_box_areas(boxes: &Array2<f64>) -> Array1<f64> {
    let num_boxes = boxes.nrows();
    let mut areas = Array1::<f64>::zeros(num_boxes);

    Zip::indexed(&mut areas).par_for_each(|i, area| {
        let box1 = boxes.row(i);
        let x1 = box1[0];
        let y1 = box1[1];
        let x2 = box1[2];
        let y2 = box1[3];
        *area = (x2 - x1 + 1.) * (y2 - y1 + 1.);
    });

    return areas;
}

pub fn remove_small_boxes(boxes: &Array2<f64>, min_size: f64) -> Array2<f64> {
    let areas = box_areas(boxes);
    let keep: Vec<usize> = areas
        .indexed_iter()
        .filter(|(_, &area)| area >= min_size)
        .map(|(index, _)| index)
        .collect();
    return boxes.select(Axis(0), &keep);
}

pub fn box_convert(boxes: &Array2<f64>, in_fmt: &BoxFormat, out_fmt: &BoxFormat) -> Array2<f64> {
    let num_boxes: usize = boxes.nrows();
    let mut converted_boxes = Array2::<f64>::zeros((num_boxes, 4));

    Zip::indexed(converted_boxes.rows_mut()).for_each(|i, mut box1| {
        let box2 = boxes.row(i);
        match (in_fmt, out_fmt) {
            (BoxFormat::XYXY, BoxFormat::XYWH) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let x2 = box2[2];
                let y2 = box2[3];
                box1[0] = x1;
                box1[1] = y1;
                box1[2] = x2 - x1 + 1.;
                box1[3] = y2 - y1 + 1.;
            }
            (BoxFormat::XYXY, BoxFormat::CXCYWH) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let x2 = box2[2];
                let y2 = box2[3];
                box1[0] = (x1 + x2) / 2.;
                box1[1] = (y1 + y2) / 2.;
                box1[2] = x2 - x1 + 1.;
                box1[3] = y2 - y1 + 1.;
            }
            (BoxFormat::XYWH, BoxFormat::XYXY) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let w = box2[2];
                let h = box2[3];
                box1[0] = x1;
                box1[1] = y1;
                box1[2] = x1 + w - 1.;
                box1[3] = y1 + h - 1.;
            }
            (BoxFormat::XYWH, BoxFormat::CXCYWH) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let w = box2[2];
                let h = box2[3];
                box1[0] = x1 + (w - 1.) / 2.;
                box1[1] = y1 + (h - 1.) / 2.;
                box1[2] = w;
                box1[3] = h;
            }
            (BoxFormat::CXCYWH, BoxFormat::XYXY) => todo!(),
            (BoxFormat::CXCYWH, BoxFormat::XYWH) => todo!(),
            (BoxFormat::XYXY, BoxFormat::XYXY) => (),
            (BoxFormat::XYWH, BoxFormat::XYWH) => (),
            (BoxFormat::CXCYWH, BoxFormat::CXCYWH) => (),
        }
    });
    return converted_boxes;
}

pub fn parallel_box_convert(
    boxes: &Array2<f64>,
    in_fmt: &BoxFormat,
    out_fmt: &BoxFormat,
) -> Array2<f64> {
    let num_boxes: usize = boxes.nrows();
    let mut converted_boxes = Array2::<f64>::zeros((num_boxes, 4));

    Zip::indexed(converted_boxes.rows_mut()).par_for_each(|i, mut box1| {
        let box2 = boxes.row(i);
        match (in_fmt, out_fmt) {
            (BoxFormat::XYXY, BoxFormat::XYWH) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let x2 = box2[2];
                let y2 = box2[3];
                box1[0] = x1;
                box1[1] = y1;
                box1[2] = x2 - x1 + 1.;
                box1[3] = y2 - y1 + 1.;
            }
            (BoxFormat::XYXY, BoxFormat::CXCYWH) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let x2 = box2[2];
                let y2 = box2[3];
                box1[0] = (x1 + x2) / 2.;
                box1[1] = (y1 + y2) / 2.;
                box1[2] = x2 - x1 + 1.;
                box1[3] = y2 - y1 + 1.;
            }
            (BoxFormat::XYWH, BoxFormat::XYXY) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let w = box2[2];
                let h = box2[3];
                box1[0] = x1;
                box1[1] = y1;
                box1[2] = x1 + w - 1.;
                box1[3] = y1 + h - 1.;
            }
            (BoxFormat::XYWH, BoxFormat::CXCYWH) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let w = box2[2];
                let h = box2[3];
                box1[0] = x1 + (w - 1.) / 2.;
                box1[1] = y1 + (h - 1.) / 2.;
                box1[2] = w;
                box1[3] = h;
            }
            (BoxFormat::CXCYWH, BoxFormat::XYXY) => todo!(),
            (BoxFormat::CXCYWH, BoxFormat::XYWH) => todo!(),
            (BoxFormat::XYXY, BoxFormat::XYXY) => (),
            (BoxFormat::XYWH, BoxFormat::XYWH) => (),
            (BoxFormat::CXCYWH, BoxFormat::CXCYWH) => (),
        }
    });
    return converted_boxes;
}

// pub fn validate_boxes(boxes: &Array2<f64>, fmt: &BoxFormat) -> bool {
//     let mut valid = true;

//     Zip::indexed(boxes.rows()).for_each(|_, box1| {
//         let x1 = box1[0];
//         let y1 = box1[1];
//         let x2 = box1[2];
//         let y2 = box1[3];
//         match fmt {
//             BoxFormat::XYXY => {
//                 if x1 > x2 || y1 > y2 {
//                     valid = false;
//                 }
//             }
//             BoxFormat::XYWH => {
//                 if x2 < x1 || y2 < y1 {
//                     valid = false;
//                 }
//             }
//             BoxFormat::CXCYWH => {
//                 if x2 < x1 || y2 < y1 {
//                     valid = false;
//                 }
//             }
//         }
//     });

//     return valid;
// }
