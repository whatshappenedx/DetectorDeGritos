import torch
from flask import Flask, render_template, jsonify, request, send_from_directory
from ModelEval import process_file
from RecordSound import record_sound
import os

# Inicia la aplicación Flask
app = Flask(__name__)

# Carga un modelo de machine learning almacenado en disco usando PyTorch
model = torch.load('Models/Resnet34_Model_2023-10-13--17-11-18.pt', map_location=torch.device('cpu'))

# Define una ruta para servir la página HTML principal
@app.route('/')
def index():
    return render_template('index.html')

# Define una ruta para iniciar una grabación de audio
@app.route('/start_record', methods=['POST'])
def start_record():
    record_sound(44100, 10)  # Graba audio con una tasa de muestreo de 44100 durante 10 segundos
    return 'Recording Completed'

# Define una ruta para evaluar la voz grabada utilizando el modelo cargado
@app.route('/evaluate_voice', methods=['POST'])
def evaluate_voice():
    evaluation_result = process_file('SoundRecord/recorded.wav', model)
    evaluation_result = bool(evaluation_result)  # Convierte el resultado de NumPy a booleano de Python
    return jsonify({'result': evaluation_result})  # Devuelve el resultado como JSON

# Define una ruta para manejar la subida y evaluación de archivos de audio
@app.route('/upload_and_evaluate', methods=['POST'])
def upload_and_evaluate():
    if 'file' not in request.files:
        return jsonify({'result': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'result': 'No selected file'}), 400
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        evaluation_result = process_file(file_path, model)
        evaluation_result = bool(evaluation_result)  # Convierte el resultado de NumPy a booleano de Python
        return jsonify({'result': evaluation_result})  # Devuelve el resultado como JSON

# Define una ruta para servir imágenes estáticas desde una carpeta de imágenes
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('images', filename)  # Envía el archivo solicitado desde la carpeta 'images'

# Ejecuta la aplicación Flask en modo de depuración
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
