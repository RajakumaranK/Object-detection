import os
from flask import Flask, request, jsonify, render_template_string
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Initialize the Flask app
app = Flask(__name__)

# Upload folder for saving images temporarily
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the pre-trained model once when the application starts
model = MobileNetV2(weights='imagenet')

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# HTML template with magenta container
html_template = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Image Upload and Prediction</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background-color: #007BFF;  /* Blue background color */
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      margin: 0;
    }
    .container {
      background-color: #FF00FF;  /* Magenta color */
      padding: 30px;
      border-radius: 8px;
      box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
      text-align: center;
      width: 400px;
    }
    h2 {
      color: #333;
      margin-bottom: 20px;
    }
    form {
      margin-bottom: 20px;
    }
    input[type="file"] {
      padding: 10px;
      border: 1px solid #ddd;
      border-radius: 5px;
      margin-bottom: 20px;
    }
    button {
      background-color: #4CAF50;
      color: white;
      padding: 15px 32px;
      font-size: 16px;
      border: none;
      border-radius: 5px;
      cursor: pointer;
    }
    button:hover {
      background-color: #45a049;
    }
    .result {
      margin-top: 30px;
      padding: 5px;
      background-color: #f0f0f0;
      border: 1px solid #ddd;
      border-radius: 5px;
      font-size: 18px;
      font-weight: bold;
      color: #333;
    }
  </style>
</head>
<body>

  <div class="container">
    <h2>Car Image Detection</h2>
    <form action="/upload" method="POST" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required>
      <br>
      <button type="submit">Upload Image</button>
    </form>

    {% if result %}
      <div class="result">
         <strong>{{ result }}</strong>
      </div>
    {% endif %}
  </div>

</body>
</html>
'''

@app.route('/')
def index():
    """Render the upload form"""
    return render_template_string(html_template)

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle the image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file and allowed_file(file.filename):
        # Securely save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Open the image file
        img = Image.open(filepath)

        # Convert the image to RGB if it has an alpha channel (RGBA)
        img = img.convert('RGB')

        # Preprocess the image for the model
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction using the pre-trained model
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # Check if any of the top predictions are 'car'
        car_detected = any("car" in label[1].lower() for label in decoded_predictions)

        result = "Car" if car_detected else "Not Car"
        return render_template_string(html_template, result=result)

    return jsonify({"error": "Invalid file format. Please upload an image."})

if __name__ == '__main__':
    # Run the app with reloader enabled (default is True when in debug mode)
    app.run(debug=True, use_reloader=True)
