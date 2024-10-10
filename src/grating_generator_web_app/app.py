# BACKEND FLASK

from flask import Flask, render_template, request, send_file
from image_generator import generate_image, generate_gabor
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_image', methods=['POST'])
def generate_image_route():
    # Obtén el valor de la barra de desplazamiento y asegurar valor flotante
    value = float(request.json.get('scrollValue', 50))
    
    # Genera la imagen en base al valor recibido
    img = generate_gabor(value)
    
    # Convierte la imagen en un stream para enviarla al frontend
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)

