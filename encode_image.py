import base64
import cv2
from PIL import Image

def encode_to_b64(img_path: str):
    img = cv2.imread(img_path)
    img_bytes = img.tobytes()
    img_b64 = base64.b64encode(img_bytes)
    return img_b64