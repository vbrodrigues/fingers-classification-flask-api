from encode_image import encode_to_b64
from flask import jsonify
import requests
import pprint
import time
import pathlib

for i, path in enumerate(pathlib.Path(r"C:\dev\Datasets\fingers\test").iterdir()):
    if i >= 5:
        break

    print(f"\nImage file: {path.name}")
    img_path = str(path)
    img = encode_to_b64(img_path)
    data = {"img": str(img)}
    t = time.time()
    response = requests.post("http://127.0.0.1:5000/finger-recognition", json=data)
    print(f"\nTime taken: {(time.time() - t):.2f} seconds.\n\nResponse:")
    pprint.PrettyPrinter().pprint(response.json())
    print("-------------------------------------------------")
