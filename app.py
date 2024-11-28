from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import os
from optimized_virtual_fitting import OptimizedVirtualFitting

app = Flask(__name__)

@app.route('/process_video', methods=['POST'])
def process_video():
    # Verificar que los archivos de video y modelo 3D estén presentes
    if 'video' not in request.files or 'garment' not in request.files:
        return {'error': 'Archivos no encontrados'}, 400
    
    video = request.files['video']
    garment = request.files['garment']
    
    # Guardar archivos temporalmente
    video_path = os.path.join('uploads', secure_filename(video.filename))
    garment_path = os.path.join('uploads', secure_filename(garment.filename))
    
    video.save(video_path)
    garment.save(garment_path)
    
    # Procesar video
    output_path = os.path.join('outputs', f'processed_{video.filename}')
    
    fitting = OptimizedVirtualFitting()
    fitting.process_video(video_path, garment_path, output_path)
    
    # Devolver el archivo procesado
    return send_file(output_path, mimetype='video/mp4')

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    app.run(host='localhost', port=5000)
    #app.run(host='0.0.0.0', port=5000)