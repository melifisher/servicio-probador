import trimesh
import numpy as np
from pathlib import Path
import glb
from pytorch3d.io import load_obj, load_ply
import open3d as o3d

class Model3DLoader:
    def __init__(self):
        self.supported_formats = {
            '.obj': self._load_obj,
            '.fbx': self._load_fbx,
            '.glb': self._load_glb,
            '.gltf': self._load_gltf,
            '.ply': self._load_ply,
            '.stl': self._load_stl,
            '.dae': self._load_collada
        }
    
    def load_model(self, file_path):
        """
        Carga un modelo 3D en cualquier formato soportado y lo convierte
        a un formato estándar (vértices y caras)
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension not in self.supported_formats:
            raise ValueError(f"Formato no soportado: {extension}")
            
        return self.supported_formats[extension](file_path)
    
    def _load_obj(self, file_path):
        """
        Carga archivos .obj
        Ventajas: 
        - Formato muy común y bien soportado
        - Bueno para modelos estáticos
        - Soporta texturas y materiales
        """
        verts, faces, aux = load_obj(file_path)
        return {
            'vertices': verts,
            'faces': faces.verts_idx,
            'textures': aux.texture_images if hasattr(aux, 'texture_images') else None
        }
    
    def _load_fbx(self, file_path):
        """
        Carga archivos .fbx
        Ventajas:
        - Excelente para animaciones
        - Mantiene jerarquía de objetos
        - Soporta rigging y skinning
        """
        try:
            import fbx
            sdk_manager = fbx.FbxManager.Create()
            scene = fbx.FbxScene.Create(sdk_manager, "")
            importer = fbx.FbxImporter.Create(sdk_manager, "")
            
            importer.Initialize(str(file_path))
            importer.Import(scene)
            
            # Extraer datos del modelo
            root_node = scene.GetRootNode()
            vertices, faces = self._process_fbx_node(root_node)
            
            return {
                'vertices': vertices,
                'faces': faces,
                'animations': self._extract_animations(scene)
            }
        except ImportError:
            raise ImportError("Se requiere el SDK de FBX para cargar archivos .fbx")
    
    def _load_glb(self, file_path):
        """
        Carga archivos .glb (Binary GLTF)
        Ventajas:
        - Formato moderno y eficiente
        - Bueno para web y móvil
        - Archivo único y compacto
        """
        scene = trimesh.load(str(file_path))
        
        if isinstance(scene, trimesh.Scene):
            # Combinar todas las geometrías en una sola malla
            vertices = []
            faces = []
            offset = 0
            
            for geometry in scene.geometry.values():
                vertices.extend(geometry.vertices)
                faces.extend(geometry.faces + offset)
                offset += len(geometry.vertices)
                
            return {
                'vertices': np.array(vertices),
                'faces': np.array(faces)
            }
        else:
            return {
                'vertices': scene.vertices,
                'faces': scene.faces
            }
    
    def _load_gltf(self, file_path):
        """
        Carga archivos .gltf
        Ventajas:
        - Formato moderno y versátil
        - Bueno para streaming
        - Soporta animaciones y materiales PBR
        """
        # Similar a GLB pero con archivos externos
        return self._load_glb(file_path)
    
    def _load_ply(self, file_path):
        """
        Carga archivos .ply
        Ventajas:
        - Simple y directo
        - Bueno para nubes de puntos
        - Soporta propiedades personalizadas
        """
        mesh = o3d.io.read_triangle_mesh(str(file_path))
        return {
            'vertices': np.asarray(mesh.vertices),
            'faces': np.asarray(mesh.triangles)
        }
    
    def _load_stl(self, file_path):
        """
        Carga archivos .stl
        Ventajas:
        - Ideal para impresión 3D
        - Simple y robusto
        - Bueno para modelos sólidos
        """
        mesh = trimesh.load_mesh(str(file_path))
        return {
            'vertices': mesh.vertices,
            'faces': mesh.faces
        }
    
    def _load_collada(self, file_path):
        """
        Carga archivos .dae (Collada)
        Ventajas:
        - Buen soporte para animaciones
        - Formato XML legible
        - Compatible con muchas herramientas
        """
        mesh = trimesh.load(str(file_path))
        if isinstance(mesh, trimesh.Scene):
            combined_mesh = trimesh.util.concatenate(
                [trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                 for g in mesh.geometry.values()]
            )
            return {
                'vertices': combined_mesh.vertices,
                'faces': combined_mesh.faces
            }
        return {
            'vertices': mesh.vertices,
            'faces': mesh.faces
        }
    
    def _process_fbx_node(self, node):
        """Procesa un nodo FBX y extrae la geometría"""
        vertices = []
        faces = []
        
        if node.GetNodeAttribute() and \
           node.GetNodeAttribute().GetAttributeType() == fbx.FbxNodeAttribute.eMesh:
            mesh = node.GetNodeAttribute()
            
            # Extraer vértices
            control_points = mesh.GetControlPoints()
            vertices.extend([(point[0], point[1], point[2]) for point in control_points])
            
            # Extraer caras
            for i in range(mesh.GetPolygonCount()):
                face = []
                for j in range(3):  # Asumimos triángulos
                    control_point_index = mesh.GetPolygonVertex(i, j)
                    face.append(control_point_index)
                faces.append(face)
        
        # Procesar nodos hijos recursivamente
        for i in range(node.GetChildCount()):
            child_verts, child_faces = self._process_fbx_node(node.GetChild(i))
            if child_verts:
                offset = len(vertices)
                vertices.extend(child_verts)
                faces.extend([[idx + offset for idx in face] for face in child_faces])
        
        return np.array(vertices), np.array(faces)
    
    def _extract_animations(self, scene):
        """Extrae datos de animación de una escena FBX"""
        animations = []
        for i in range(scene.GetSrcObjectCount(fbx.FbxAnimStack.ClassId)):
            anim_stack = scene.GetSrcObject(fbx.FbxAnimStack.ClassId, i)
            animations.append({
                'name': anim_stack.GetName(),
                'start': anim_stack.LocalStart.Get().GetSecondDouble(),
                'end': anim_stack.LocalStop.Get().GetSecondDouble()
            })
        return animations

# Ejemplo de uso
def load_model_example():
    loader = Model3DLoader()
    
    # Cargar diferentes formatos
    obj_model = loader.load_model("model.obj")
    fbx_model = loader.load_model("model.fbx")
    glb_model = loader.load_model("model.glb")
    
    # Usar el modelo cargado
    vertices = obj_model['vertices']
    faces = obj_model['faces']
    
    return vertices, faces