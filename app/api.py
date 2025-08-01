import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from app.model_loader import load_model, predict_image
from flask_cors import CORS

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 👇 Correct Flask setup to serve static index.html
app = Flask(__name__, static_folder='static', static_url_path='/')
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load model once at startup
model = load_model()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ✅ Serve the index.html as homepage
@app.route('/')
def index():
    return app.send_static_file('index.html')

# ✅ Handle prediction request
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            prediction = predict_image(filepath, model)
            os.remove(filepath)  # cleanup after prediction
            return jsonify({'prediction': prediction})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file format'}), 400
