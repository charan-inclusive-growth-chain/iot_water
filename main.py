from flask import Flask, request, redirect, jsonify
from iot_image_processing import *
app = Flask(__name__)


@app.route('/iot-image-processing', methods=['POST'])
def upload_file():
    content = request.get_json()
    path = content["img_path"]
    return main(path)


if __name__ == "__main__":
    app.run()