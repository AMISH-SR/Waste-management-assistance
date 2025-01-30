from flask import Flask, request, jsonify, render_template
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)

model = MobileNetV2(weights='imagenet')

def classify_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    categories = {
        'plastic': 'Recyclable', 'paper': 'Recyclable', 'bottle': 'Recyclable','jacket':'Recyclable',
        'cardboard': 'Recyclable', 'syringe': 'Composable', 'organic': 'Composable','binder':'Recyclable',
        'envelope': 'Recyclable', 'file': 'Recyclable', 'metal': 'Recyclable',
        'glass': 'Recyclable', 'electronics': 'E-Waste', 'carton': 'Recyclable',
        'crate': 'Recyclable', 'other': 'General Waste'
    }

    results = []
    for _, label, _ in decoded_predictions:
        label_lower = label.lower()
        if 'bottle' in label_lower:
            category = 'Recyclable'
        elif 'jacket' in label_lower:
            category = 'Recyclabe'
        else:
            category = categories.get(label_lower, 'General Waste')
        results.append({"label": label, "category": category})

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        img = Image.open(file)
        results = classify_image(img)
        return jsonify({"status": "success", "results": results})
    return jsonify({"status": "error", "message": "No file uploaded"})

if __name__ == '__main__':
    app.run(debug=True)