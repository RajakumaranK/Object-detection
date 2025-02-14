import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Upload folder for saving images temporarily
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load OpenCV's pre-trained Haar Cascade for face detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to detect human faces using Haar cascade
def detect_human(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If at least one face is detected, classify as "Human"
    return "Human" if len(faces) > 0 else "Not Human"

# HTML Template (Same as before)
html_template = ''' 
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Human Detection</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #007BFF; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
    .container { background-color: #FF00FF; padding: 30px; border-radius: 8px; box-shadow: 0 0 15px rgba(0, 0, 0, 0.1); text-align: center; width: 400px; }
    h2 { color: #333; margin-bottom: 20px; }
    form { margin-bottom: 20px; }
    input[type="file"] { padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px; }
    button { background-color: #4CAF50; color: white; padding: 15px 32px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer; }
    button:hover { background-color: #45a049; }
    .result { margin-top: 30px; padding: 5px; background-color: #f0f0f0; border: 1px solid #ddd; border-radius: 5px; font-size: 18px; font-weight: bold; color: #333; }
  </style>
</head>
<body>

  <div class="container">
    <h2>Human Detection</h2>
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

        # Detect if the image contains a human
        result = detect_human(filepath)

        return render_template_string(html_template, result=result)

    return jsonify({"error": "Invalid file format. Please upload an image."})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
