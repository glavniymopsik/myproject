from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO('best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    filename = f"{uuid.uuid4().hex}.jpg"
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(upload_path)

    # Обработка изображения
    results = model(upload_path)

    # Сохраняем результат в отдельную папку
    result_subdir = os.path.join(RESULT_FOLDER, filename[:-4])
    os.makedirs(result_subdir, exist_ok=True)
    result_image_path = os.path.join(result_subdir, 'result.jpg')
    results[0].save(filename=result_image_path)

    # Возвращаем относительный путь для отображения в браузере
    result_img_url = f'/results/{filename[:-4]}/result.jpg'
    return jsonify({'result': result_img_url})

# Обслуживаем файлы из папки results
@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True)
