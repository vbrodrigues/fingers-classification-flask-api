from flask import Flask, request, render_template, jsonify 
from flask_cors import CORS
from predict import predict
from PIL import Image
import base64
import numpy as np
import matplotlib.pyplot as plt
import uuid
import datetime

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    try:
        return jsonify({"success": True, "message": "Request received.", "created_at": datetime.datetime.now().strftime("%D %T")})
    except Exception as e:
        print(e)
        return jsonify({"success": False, "message": e}), 500    

@app.route("/finger-recognition", methods = ["POST"])
def pred():
    try:
        data = request.get_json()
        from_b64 = base64.b64decode(data["img"][2:-1].encode())
        img = Image.frombytes("RGB", (128, 128), from_b64)
        img.save(f"data/{int(uuid.uuid1())}.png", format="png")
        prediction = predict(img)
        return jsonify({"success": True, "prediction": prediction, "created_at": datetime.datetime.now().strftime("%D %T")})

    except Exception as e:
        print(e)
        return jsonify({"success": False, "message": e}), 500  

if __name__ == "__main__":
    app.run(debug = True)