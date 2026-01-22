import logging
import cv2

logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s.%(msecs)03d|%(levelname)s|%(filename)s:%(lineno)d|%(funcName)s -> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

def resize_img(input_image):
    (h, w, ch) = input_image.shape
    d = max(w, h)
    scale_to = 640 if d >= 1280 else d / 2
    scale_to = max(64, scale_to)
    input_scale = d / scale_to
    output_image = cv2.resize(input_image, (int(w / input_scale), int(h / input_scale)), interpolation=cv2.INTER_LINEAR)
    logging.debug(f"resize image from: {(h, w, ch)} -> {output_image.shape}")
    return output_image, input_scale