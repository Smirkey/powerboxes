# import os

# import numpy as np
# from PIL import Image
# from powerboxes import masks_to_boxes


# def test_masks_box():
#     expected = np.array(
#         [
#             [127, 2, 165, 40],
#             [2, 50, 44, 92],
#             [56, 63, 98, 100],
#             [139, 68, 175, 104],
#             [160, 112, 198, 145],
#             [49, 138, 99, 182],
#             [108, 148, 152, 213],
#         ],
#     )
#     assets_directory = os.path.join(
#         os.path.dirname(os.path.abspath(__file__)), "assets"
#     )
#     mask_path = os.path.join(assets_directory, "masks.tiff")
#     image = Image.open(mask_path)
#     masks = np.zeros((image.n_frames, image.height, image.width))
#     for index in range(image.n_frames):
#         image.seek(index)
#         masks[index] = np.array(image)
#     out = masks_to_boxes(masks.astype(np.bool_))
#     assert out.dtype == np.dtype("uint64")
#     np.testing.assert_allclose(out, expected, atol=1e-4)
