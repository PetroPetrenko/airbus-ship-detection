import numpy as np
import cv2

def rle_decode(mask_rle, shape):
    """
    Decode a run-length encoded mask into a binary mask.
    """
    if pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)

    s = np.fromstring(mask_rle, sep=' ', dtype=np.int)
    starts, lengths = s[::2] - 1, s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for start, end in zip(starts, ends):
        img[start:end] = 1

    return img.reshape(shape).T  # Needed to align to the original shape

def rle_encode(mask):
    """
    Encode a binary mask into run-length encoding.
    """
    pixels = mask.flatten()
    rle = []
    last_pixel = -1
    for i, pixel in enumerate(pixels):
        if pixel != last_pixel:
            rle.append(i + 1)
            rle.append(1)
            last_pixel = pixel
        else:
            rle[-1] += 1
    return ' '.join(map(str, rle)) if rle else ''
