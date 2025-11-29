from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import base64

app = Flask(__name__)

# Load your trained model
model = load_model('casting_product_detection.keras')

# Directory to store uploaded or captured images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_path = None

    # Check if user uploaded a file
    if 'file' in request.files and request.files['file'].filename != '':
        img_file = request.files['file']
        image_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
        img_file.save(image_path)
    else:
        # If no file uploaded, check for camera input
        data_url = request.form.get('camera_image')
        if data_url:
            header, encoded = data_url.split(",", 1)
            data = base64.b64decode(encoded)
            image_path = os.path.join(UPLOAD_FOLDER, "captured_image.png")
            with open(image_path, "wb") as f:
                f.write(data)
        else:
            return render_template('index.html', prediction="No image provided!")

    # Preprocess image for model
    img = Image.open(image_path).convert('L').resize((300, 300))  # grayscale
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # shape (1,300,300,1)

    # Predict
    prediction = model.predict(img_array)
    result = "Defective Casting ❌" if prediction[0][0] > 0.5 else "OK Casting ✅"

    return render_template('index.html', prediction=result, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
