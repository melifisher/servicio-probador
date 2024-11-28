import numpy as np
from scipy.spatial.transform import Rotation
import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

class GarmentTransform3D:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def calculate_body_measurements(self, landmarks):
        """
        Calcula las medidas principales del cuerpo usando los landmarks
        """
        # Obtener puntos clave relevantes
        shoulders = np.array([
            [landmarks[11].x, landmarks[11].y, landmarks[11].z],  # Hombro izquierdo
            [landmarks[12].x, landmarks[12].y, landmarks[12].z]   # Hombro derecho
        ])
        
        hips = np.array([
            [landmarks[23].x, landmarks[23].y, landmarks[23].z],  # Cadera izquierda
            [landmarks[24].x, landmarks[24].y, landmarks[24].z]   # Cadera derecha
        ])
        
        # Calcular medidas
        shoulder_width = np.linalg.norm(shoulders[1] - shoulders[0])
        hip_width = np.linalg.norm(hips[1] - hips[0])
        torso_height = np.linalg.norm(
            np.mean(shoulders, axis=0) - np.mean(hips, axis=0)
        )
        
        return {
            'shoulder_width': shoulder_width,
            'hip_width': hip_width,
            'torso_height': torso_height
        }
    
    def estimate_body_orientation(self, landmarks):
        """
        Estima la orientación del cuerpo en 3D
        """
        # Calcular vector frontal usando hombros
        shoulder_vector = np.array([
            landmarks[12].x - landmarks[11].x,
            landmarks[12].y - landmarks[11].y,
            landmarks[12].z - landmarks[11].z
        ])
        
        # Vector hacia arriba usando la columna vertebral
        spine_vector = np.array([
            landmarks[23].x - landmarks[11].x,
            landmarks[23].y - landmarks[11].y,
            landmarks[23].z - landmarks[11].z
        ])
        
        # Normalizar vectores
        shoulder_vector = shoulder_vector / np.linalg.norm(shoulder_vector)
        spine_vector = spine_vector / np.linalg.norm(spine_vector)
        
        # Calcular vector lateral (producto cruz)
        side_vector = np.cross(shoulder_vector, spine_vector)
        side_vector = side_vector / np.linalg.norm(side_vector)
        
        # Crear matriz de rotación
        rotation_matrix = np.stack([shoulder_vector, spine_vector, side_vector])
        
        return rotation_matrix
    
    def deform_garment_mesh(self, vertices, faces, measurements, orientation):
        """
        Deforma la malla de la prenda para ajustarse a las medidas del cuerpo
        """
        # Convertir a tensor de PyTorch
        vertices_torch = torch.tensor(vertices, device=self.device)
        faces_torch = torch.tensor(faces, device=self.device)
        
        # Calcular factores de escala
        original_measurements = self.calculate_mesh_measurements(vertices)
        scale_factors = {
            'width': measurements['shoulder_width'] / original_measurements['width'],
            'height': measurements['torso_height'] / original_measurements['height']
        }
        
        # Aplicar escala
        scaled_vertices = vertices_torch.clone()
        scaled_vertices[:, 0] *= scale_factors['width']
        scaled_vertices[:, 1] *= scale_factors['height']
        
        # Aplicar rotación
        rotation_torch = torch.tensor(orientation, device=self.device)
        rotated_vertices = torch.matmul(scaled_vertices, rotation_torch)
        
        # Crear malla deformada
        deformed_mesh = Meshes(
            verts=[rotated_vertices],
            faces=[faces_torch]
        )
        
        return deformed_mesh
    
    def apply_physical_constraints(self, mesh, landmarks):
        """
        Aplica restricciones físicas para evitar intersecciones y mantener la forma natural
        """
        # Convertir landmarks a tensor
        body_points = torch.tensor(
            [[l.x, l.y, l.z] for l in landmarks],
            device=self.device
        )
        
        # Muestrear puntos de la superficie de la malla
        sample_points = sample_points_from_meshes(mesh, num_samples=1000)
        
        # Calcular distancia Chamfer entre la malla y los puntos del cuerpo
        loss, _ = chamfer_distance(sample_points, body_points.unsqueeze(0))
        
        # Aplicar gradientes para ajustar la malla (esto requeriría optimización iterativa)
        # Este es un placeholder para la lógica completa de optimización
        adjusted_verts = mesh.verts_packed() - 0.1 * loss.gradient()
        
        return Meshes(
            verts=[adjusted_verts],
            faces=[mesh.faces_packed()]
        )
    
    def calculate_mesh_measurements(self, vertices):
        """
        Calcula las medidas básicas de la malla
        """
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        dimensions = max_coords - min_coords
        
        return {
            'width': dimensions[0],
            'height': dimensions[1],
            'depth': dimensions[2]
        }
    
    def transform_garment(self, vertices, faces, landmarks):
        """
        Función principal que combina todas las transformaciones
        """
        # Calcular medidas del cuerpo
        measurements = self.calculate_body_measurements(landmarks)
        
        # Estimar orientación
        orientation = self.estimate_body_orientation(landmarks)
        
        # Deformar malla inicial
        deformed_mesh = self.deform_garment_mesh(vertices, faces, measurements, orientation)
        
        # Aplicar restricciones físicas
        final_mesh = self.apply_physical_constraints(deformed_mesh, landmarks)
        
        return final_mesh.verts_packed().cpu().numpy(), final_mesh.faces_packed().cpu().numpy()

# Ejemplo de uso
def transform_garment_example():
    transformer = GarmentTransform3D()
    
    # Supongamos que tenemos estos datos
    vertices = np.random.rand(100, 3)  # Vértices de la prenda
    faces = np.random.randint(0, 100, (50, 3))  # Caras de la prenda
    landmarks = [...]  # Landmarks del cuerpo desde MediaPipe
    
    # Aplicar transformación
    transformed_verts, transformed_faces = transformer.transform_garment(
        vertices, faces, landmarks
    )
    
    return transformed_verts, transformed_faces