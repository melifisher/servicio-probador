import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from PIL import Image
import onnxruntime as ort
from scipy.spatial.transform import Rotation
import threading
from queue import Queue
import time
from concurrent.futures import ThreadPoolExecutor

class OptimizedVirtualFitting:
    def __init__(self):
        #self.model_loader = Model3DLoader()
        self.pose_detector = self._initialize_pose_detector()
        self.frame_queue = Queue(maxsize=30)  # Buffer para frames
        self.result_queue = Queue(maxsize=30)  # Buffer para resultados
        
    def _initialize_pose_detector(self):
        """
        Inicializa MediaPipe optimizado para CPU
        """
        mp_pose = mp.solutions.pose
        return mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Usar modelo más ligero
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True,  # Enable landmark smoothing
            smooth_segmentation=True  # Enable segmentation smoothing
        )
    
    def optimize_mesh(self, vertices, faces):
        """
        Optimiza la malla para procesamiento en CPU
        """
        # Reducir número de vértices si es necesario
        if len(vertices) > 5000:  # Umbral ajustable
            vertices, faces = self._decimate_mesh(vertices, faces)
            
        # Convertir a tipos de datos más eficientes
        return vertices.astype(np.float32), faces.astype(np.int32)
    
    def _decimate_mesh(self, vertices, faces, target_ratio=0.5):
        """
        Reduce la complejidad de la malla
        """
        import open3d as o3d
        
        # Convertir a formato Open3D
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # Decimación
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=int(len(faces) * target_ratio)
        )
        
        return np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    
    def process_frame_batch(self, frames):
        """
        Procesa múltiples frames en paralelo usando threading
        """
        print("DENTRO DEL PROCESS FRAME BATCH")
        #frames_list = list(frames)
        # Normalize frame timestamps
        #normalized_frames = self._normalize_frame_timestamps(frames)
        
        normalized_frames = [
            cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA) 
            for frame in frames
        ]
        results = []
        # threads = []
        
        # for frame in frames:
        #     thread = threading.Thread(
        #         target=lambda q, f: q.put(self._process_single_frame(f)),
        #         args=(self.result_queue, frame)
        #     )
        #     threads.append(thread)
        #     thread.start()
        
        # # Esperar a que todos los threads terminen
        # for thread in threads:
        #     thread.join()
        #     results.append(self.result_queue.get())
        
        # with ThreadPoolExecutor(max_workers=5) as executor:
        #     # Map processing to threads with better timestamp management
        #     results = list(executor.map(self._process_single_frame, normalized_frames))

        # Process frames sequentially to ensure timestamp consistency
        for frame in normalized_frames:
            try:
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                result = self.pose_detector.process(frame_rgb)
                results.append(result)
            
            except Exception as conversion_error:
                print(f"Frame conversion error: {conversion_error}")
                results.append(None)

        return results
    
    def _normalize_frame_timestamps(self, frames):
        """
        Normalize frame timestamps to ensure sequential processing
        """
        # Create a copy to avoid modifying original frames
        normalized_frames = frames.copy()
        
        # Resize frames to a consistent size
        target_size = (640, 480)  # Standard HD resolution
        normalized_frames = [
            cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA) 
            for frame in normalized_frames
        ]

        return normalized_frames

    def _process_single_frame(self, frame):
        """
        Procesa un único frame con optimizaciones
        """
        try:
            print("DENTRO DE PROCESS SIGLE FRAME")
            # Ensure consistent frame size and format
            frame_rgb = cv2.resize(frame, (640, 480))  # Standardize resolution
            # Convertir a RGB y reducir resolución si es necesario
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            

            results = self.pose_detector.process(frame_rgb)
            return results

        except Exception as conversion_error:
            print(f"Frame conversion error: {conversion_error}")
            return None
    
    def apply_garment(self, frame, garment_data, pose_landmarks):
        """
        Aplica la prenda al frame de manera optimizada
        """
        vertices = garment_data['vertices']
        faces = garment_data['faces']
        
        # Optimizar malla si es necesario
        vertices, faces = self.optimize_mesh(vertices, faces)
        
        # Transformación básica
        transformed_verts = self._transform_vertices(vertices, pose_landmarks)
        
        # Renderizado optimizado
        return self._render_optimized(frame, transformed_verts, faces)
    
    def _transform_vertices(self, vertices, landmarks):
        """
        Transformación simplificada y optimizada de vértices
        """
        # Obtener puntos clave del cuerpo
        shoulder_left = np.array([
            landmarks[11].x,
            landmarks[11].y,
            landmarks[11].z
        ])
        shoulder_right = np.array([
            landmarks[12].x,
            landmarks[12].y,
            landmarks[12].z
        ])
        hip_left = np.array([
            landmarks[23].x,
            landmarks[23].y,
            landmarks[23].z
        ])
        
        # Calcular transformación básica
        scale = np.linalg.norm(shoulder_right - shoulder_left)
        rotation = Rotation.align_vectors(
            [shoulder_right - shoulder_left],
            [[1, 0, 0]]
        )[0]
        
        # Aplicar transformación
        transformed = vertices * scale
        transformed = rotation.apply(transformed)
        
        # Ajustar posición
        center = (shoulder_left + shoulder_right) / 2
        transformed += center
        
        return transformed
    
    def _render_optimized(self, frame, vertices, faces):
        """
        Renderizado optimizado para CPU
        """
        # Crear máscara vacía
        mask = np.zeros_like(frame)
        
        # Proyectar vértices 3D a 2D
        vertices_2d = self._project_2d_optimized(vertices, frame.shape)
        
        # Renderizar solo las caras visibles
        visible_faces = self._get_visible_faces(vertices, faces)
        
        for face in visible_faces:
            points = vertices_2d[face].astype(np.int32)
            cv2.fillPoly(mask, [points], (255, 255, 255))
        
        # Combinar usando operaciones optimizadas
        return cv2.addWeighted(frame, 1, mask, 0.7, 0)
    
    def _project_2d_optimized(self, vertices_3d, frame_shape):
        """
        Proyección 2D optimizada
        """
        # Parámetros de cámara simplificados
        f = frame_shape[1]  # Distancia focal aproximada
        c_x = frame_shape[1] / 2
        c_y = frame_shape[0] / 2
        
        # Proyección vectorizada
        vertices_2d = np.zeros((len(vertices_3d), 2), dtype=np.float32)
        vertices_2d[:, 0] = vertices_3d[:, 0] * f / (vertices_3d[:, 2] + 1e-6) + c_x
        vertices_2d[:, 1] = vertices_3d[:, 1] * f / (vertices_3d[:, 2] + 1e-6) + c_y
        
        return vertices_2d
    
    def _get_visible_faces(self, vertices, faces):
        """
        Determina las caras visibles de manera eficiente
        """
        # Calcular normales de las caras
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        # Calcular normales usando producto cruz
        normals = np.cross(v1 - v0, v2 - v0)
        
        # Determinar visibilidad basada en el producto punto con vector de vista
        view_vector = np.array([0, 0, 1])
        visibility = np.dot(normals, view_vector) < 0
        
        return faces[visibility]
    
    def process_video(self, video_path, garment_path, output_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Configure output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path, 
            fourcc, 
            fps, 
            (width, height)
        )
        
        # Load garment model
        garment_data = self.load_model(garment_path)

        frame_count = 0
        processed_frames = 0
        unprocessed_frames = 0

        try:
            print("Entrando al try")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process single frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose_detector.process(frame_rgb)
                
                if results.pose_landmarks:
                    processed_frame = self.apply_garment(
                        frame,
                        garment_data,
                        results.pose_landmarks.landmark
                    )
                    out.write(processed_frame)
                    processed_frames += 1
                    print(f"Processed frames: {processed_frames}")
                else:
                    out.write(frame)
                    unprocessed_frames += 1
                    print(f"Unprocessed frames: {unprocessed_frames}")
                

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
        frame_count = processed_frames + unprocessed_frames
        print(f"Processed {processed_frames} frames out of {frame_count} total frames")
    
    def load_model(self, file_path):
        """
        Carga un modelo 3D en cualquier formato soportado y lo convierte
        a un formato estándar (vértices y caras)
        """
        print("DENTRO DE LOAD MODEL")
        # Define supported formats and their loading functions
        self.supported_formats = {
            '.obj': self._load_obj,
            '.stl': self._load_stl,
            '.ply': self._load_ply
        }
        
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension not in self.supported_formats:
            raise ValueError(f"Formato no soportado: {extension}")
        
        return self.supported_formats[extension](file_path)

    def _load_obj(self, file_path):
        """
        Carga un modelo OBJ
        """
        import trimesh
        print("DENTRO DE LOAD_OBJ")

        # Cargar la escena o malla
        mesh = trimesh.load(str(file_path), force='mesh')
        
        # Si es una escena, tomar la primera malla
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0]
        
        return {
            'vertices': mesh.vertices,
            'faces': mesh.faces
        }

    def _load_stl(self, file_path):
        """
        Carga un modelo STL
        """
        import trimesh
        
        mesh = trimesh.load(str(file_path), force='mesh')
        return {
            'vertices': mesh.vertices,
            'faces': mesh.faces
        }

    def _load_ply(self, file_path):
        """
        Carga un modelo PLY
        """
        import trimesh
        
        mesh = trimesh.load(str(file_path), force='mesh')
        return {
            'vertices': mesh.vertices,
            'faces': mesh.faces
        }

# Ejemplo de uso
if __name__ == "__main__":
    fitting = OptimizedVirtualFitting()
    
    fitting.process_video(
        video_path="persona.mp4",
        garment_path="tshirt.obj",
        output_path="output.mp4"
    )