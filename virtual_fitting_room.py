import cv2
import numpy as np
import mediapipe as mp
import torch
from pathlib import Path
from model_3d_loader import Model3DLoader
from garment_transform_3d import GarmentTransform3D
from PIL import Image
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj

class VirtualFittingRoom:
    def __init__(self):
        self.model_loader = Model3DLoader()
        self.transformer = GarmentTransform3D()
        # Inicializar MediaPipe para detección de poses
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process_garment(self, garment_path, frame, landmarks):
        """
        Procesa una prenda desde cualquier formato 3D y la ajusta al cuerpo
        """
        # Cargar el modelo 3D
        garment_data = self.model_loader.load_model(garment_path)
        
        # Extraer vértices y caras
        vertices = garment_data['vertices']
        faces = garment_data['faces']
        
        # Si hay animaciones disponibles, procesarlas
        animations = garment_data.get('animations', None)
        if animations:
            vertices = self._process_animations(vertices, animations, frame)
        
        # Transformar la prenda según los landmarks del cuerpo
        transformed_verts, transformed_faces = self.transformer.transform_garment(
            vertices, faces, landmarks
        )
        
        return {
            'vertices': transformed_verts,
            'faces': transformed_faces,
            'textures': garment_data.get('textures', None)
        }
    
    def _process_animations(self, vertices, animations, frame_number):
        """
        Procesa animaciones si están disponibles en el modelo
        """
        if not animations:
            return vertices
            
        # Encontrar la animación correspondiente al frame actual
        current_time = frame_number / 30.0  # Asumiendo 30 fps
        for anim in animations:
            if anim['start'] <= current_time <= anim['end']:
                return self._interpolate_animation(vertices, anim, current_time)
        
        return vertices
    
    def _interpolate_animation(self, vertices, animation, current_time):
        """
        Interpola los vértices según el tiempo de animación
        """
        # Este es un ejemplo simple de interpolación lineal
        # En la práctica, necesitarías implementar interpolación más compleja
        t = (current_time - animation['start']) / (animation['end'] - animation['start'])
        return vertices * (1 - t) + vertices * t  # Simplificado para el ejemplo
    
    def try_on_garment(self, video_path, garment_path, output_path):
        """
        Procesa un video completo aplicando la prenda virtual
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Configurar el escritor de video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detectar poses
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                # Procesar la prenda para este frame
                transformed_garment = self.process_garment(
                    garment_path,
                    frame_number,
                    results.pose_landmarks.landmark
                )
                
                # Renderizar la prenda sobre el frame
                frame = self._render_garment(
                    frame,
                    transformed_garment['vertices'],
                    transformed_garment['faces'],
                    transformed_garment['textures']
                )
            
            out.write(frame)
            frame_number += 1
            
            # Mostrar preview
            cv2.imshow('Virtual Fitting', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    

    def process_video(self, video_path, garment_3d_path):
        """
        Procesa el video y superpone la prenda 3D
        """
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detectar poses en el frame
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                # Obtener puntos clave del cuerpo
                landmarks = results.pose_landmarks.landmark
                
                # Cargar modelo 3D de la prenda
                verts, faces, aux = load_obj(garment_3d_path)
                
                # Ajustar la prenda al cuerpo
                transformed_verts = self.adjust_garment_to_body(verts, landmarks)
                
                # Renderizar la prenda sobre el frame
                frame = self.render_garment(frame, transformed_verts, faces)
            
            # Mostrar resultado
            cv2.imshow('Virtual Fitting Room', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    
    def adjust_garment_to_body(self, vertices, landmarks):
        """
        Ajusta el modelo 3D de la prenda según los puntos del cuerpo
        """
        # Obtener puntos clave relevantes para la prenda
        shoulders = np.array([
            [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        ])
        
        # Transformar vértices según puntos del cuerpo
        # Aquí iría la lógica de transformación 3D
        transformed_vertices = vertices  # Placeholder
        
        return transformed_vertices
    
    def render_garment(self, frame, vertices, faces):
        """
        Renderiza la prenda 3D sobre el frame
        """
        # Aquí iría la lógica de renderizado 3D
        # Se necesitaría una biblioteca de renderizado 3D como PyOpenGL o pytorch3d
        
        return frame

    def _render_garment(self, frame, vertices, faces, textures=None):
        """
        Renderiza la prenda sobre el frame
        """
        # Proyectar vértices 3D a 2D
        vertices_2d = self._project_to_2d(vertices, frame.shape)
        
        # Crear máscara para la prenda
        mask = np.zeros_like(frame)
        
        # Dibujar triángulos
        for face in faces:
            points = vertices_2d[face].astype(np.int32)
            if textures:
                # Si hay texturas, aplicarlas
                self._apply_texture(frame, points, textures[face])
            else:
                # Si no hay texturas, usar un color sólido
                cv2.fillPoly(mask, [points], (255, 255, 255))
        
        # Combinar frame original con la prenda
        return cv2.addWeighted(frame, 1, mask, 0.7, 0)
    
    def _project_to_2d(self, vertices_3d, frame_shape):
        """
        Proyecta vértices 3D a coordenadas 2D de la imagen
        """
        # Parámetros de cámara aproximados
        focal_length = frame_shape[1]  # Usar ancho de frame como aproximación
        center = (frame_shape[1] / 2, frame_shape[0] / 2)
        
        # Matriz de proyección simple
        vertices_2d = np.zeros((len(vertices_3d), 2))
        for i, vertex in enumerate(vertices_3d):
            # Proyección perspectiva simple
            if vertex[2] != 0:  # Evitar división por cero
                vertices_2d[i, 0] = (vertex[0] * focal_length / vertex[2]) + center[0]
                vertices_2d[i, 1] = (vertex[1] * focal_length / vertex[2]) + center[1]
        
        return vertices_2d
    
    def _apply_texture(self, frame, points, texture):
        """
        Aplica una textura a un triángulo de la prenda
        """
        # Crear máscara para el triángulo actual
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [points], (255, 255, 255))
        
        # Aplicar la textura solo en el área del triángulo
        texture_resized = cv2.resize(texture, (frame.shape[1], frame.shape[0]))
        frame_with_texture = cv2.addWeighted(
            frame,
            0.7,
            cv2.bitwise_and(texture_resized, mask),
            0.3,
            0
        )
        
        return frame_with_texture



# Ejemplo de uso
def main():
    fitting_system = VirtualFittingSystem()
    
    # Probar diferentes formatos de archivo
    garment_paths = [
        "garment.obj",
        "garment.fbx",
        "garment.glb",
        "garment.ply"
    ]
    
    for garment_path in garment_paths:
        fitting_system.try_on_garment(
            video_path="input_video.mp4",
            garment_path=garment_path,
            output_path=f"output_{Path(garment_path).stem}.mp4"
        )

if __name__ == "__main__":
    fitting_room = VirtualFittingRoom()
    #fitting_room.process_video(
    #    video_path="input_video.mp4",
    #    garment_3d_path="garment_model.obj"
    #)
    
    fitting_room.try_on_garment(
        video_path="persona.mp4",
        garment_path="tshirt.obj",
        output_path="resultado.mp4"
    )