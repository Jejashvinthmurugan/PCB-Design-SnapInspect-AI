import cv2
import numpy as np


def preprocess_image(image):

    image_bgr = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    image_bgr = cv2.resize(
        image_bgr,
        (640, 640)
    )

    lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    processed = cv2.merge(
        (l, a, b)
    )

    processed = cv2.cvtColor(
        processed,
        cv2.COLOR_LAB2RGB
    )

    return processed