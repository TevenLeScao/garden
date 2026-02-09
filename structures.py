"""Structure drawing classes for the Claude Garden sketch.

Structures are 3D polyhedra projected to 2D with light-based hatching.

To add a new structure type:
1. Create a new class inheriting from StructureDrawer
2. Implement the draw(x, y, size, angle, detail) method
3. Register it in STRUCTURE_REGISTRY in sketch_claude_garden.py
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import vsketch


@dataclass
class Vec3:
    """3D vector for polyhedra geometry."""
    x: float
    y: float
    z: float

    def __add__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vec3':
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other: 'Vec3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vec3') -> 'Vec3':
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self) -> 'Vec3':
        length = self.length()
        if length < 1e-10:
            return Vec3(0, 0, 1)
        return Vec3(self.x / length, self.y / length, self.z / length)


@dataclass
class Face3D:
    """A face of a polyhedron with vertex indices."""
    vertex_indices: List[int]

    def compute_normal(self, vertices: List[Vec3]) -> Vec3:
        """Compute outward-facing normal from vertices (assumes CCW winding)."""
        if len(self.vertex_indices) < 3:
            return Vec3(0, 0, 1)

        v0 = vertices[self.vertex_indices[0]]
        v1 = vertices[self.vertex_indices[1]]
        v2 = vertices[self.vertex_indices[2]]

        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = edge1.cross(edge2)
        return normal.normalize()

    def compute_centroid(self, vertices: List[Vec3]) -> Vec3:
        """Get center point of face for depth sorting."""
        if not self.vertex_indices:
            return Vec3(0, 0, 0)

        cx, cy, cz = 0.0, 0.0, 0.0
        for idx in self.vertex_indices:
            v = vertices[idx]
            cx += v.x
            cy += v.y
            cz += v.z

        n = len(self.vertex_indices)
        return Vec3(cx / n, cy / n, cz / n)


@dataclass
class ProjectedFace:
    """A 2D projected face ready for drawing."""
    points_2d: List[Tuple[float, float]]
    normal_3d: Vec3
    depth: float


class Polyhedron3D:
    """A 3D polyhedron with vertices and faces."""

    def __init__(self, vertices: List[Vec3], faces: List[Face3D]):
        self.vertices = vertices
        self.faces = faces

    def rotate_y(self, angle: float) -> 'Polyhedron3D':
        """Rotate around Y axis."""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        new_vertices = []
        for v in self.vertices:
            new_vertices.append(Vec3(
                v.x * cos_a + v.z * sin_a,
                v.y,
                -v.x * sin_a + v.z * cos_a
            ))
        return Polyhedron3D(new_vertices, self.faces)

    def rotate_x(self, angle: float) -> 'Polyhedron3D':
        """Rotate around X axis."""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        new_vertices = []
        for v in self.vertices:
            new_vertices.append(Vec3(
                v.x,
                v.y * cos_a - v.z * sin_a,
                v.y * sin_a + v.z * cos_a
            ))
        return Polyhedron3D(new_vertices, self.faces)

    def project_to_2d(self, scale: float, x_offset: float, y_offset: float) -> List[ProjectedFace]:
        """Orthographic projection to 2D, culling back faces and sorting by depth."""
        projected_faces = []

        # View direction is -Z (looking into the screen)
        view_dir = Vec3(0, 0, -1)

        for face in self.faces:
            normal = face.compute_normal(self.vertices)

            # Back-face culling: skip faces pointing away from viewer
            if normal.dot(view_dir) <= 0:
                continue

            # Project vertices to 2D (orthographic: just drop Z, flip Y)
            points_2d = []
            for idx in face.vertex_indices:
                v = self.vertices[idx]
                px = x_offset + v.x * scale
                py = y_offset - v.y * scale  # Flip Y for screen coordinates
                points_2d.append((px, py))

            # Compute depth from face centroid
            centroid = face.compute_centroid(self.vertices)
            depth = centroid.z

            projected_faces.append(ProjectedFace(points_2d, normal, depth))

        # Sort by depth (back to front for painter's algorithm)
        projected_faces.sort(key=lambda f: f.depth)

        return projected_faces


class PolyhedronGenerator:
    """Static methods to generate various polyhedra."""

    @staticmethod
    def deformed_octahedron(vsk: vsketch.Vsketch, deform: float = 0.3) -> Polyhedron3D:
        """Create octahedron with random vertex deformation."""
        vertices = [
            Vec3(0, 1, 0),    # 0: top
            Vec3(0, -1, 0),   # 1: bottom
            Vec3(1, 0, 0),    # 2: +x
            Vec3(-1, 0, 0),   # 3: -x
            Vec3(0, 0, 1),    # 4: +z
            Vec3(0, 0, -1),   # 5: -z
        ]

        # Add random perturbation
        deformed = []
        for v in vertices:
            deformed.append(Vec3(
                v.x + vsk.random(-deform, deform),
                v.y + vsk.random(-deform, deform),
                v.z + vsk.random(-deform, deform)
            ))

        # 8 triangular faces (CCW winding when viewed from outside)
        faces = [
            Face3D([0, 4, 2]),   # top front right
            Face3D([0, 2, 5]),   # top back right
            Face3D([0, 5, 3]),   # top back left
            Face3D([0, 3, 4]),   # top front left
            Face3D([1, 2, 4]),   # bottom front right
            Face3D([1, 5, 2]),   # bottom back right
            Face3D([1, 3, 5]),   # bottom back left
            Face3D([1, 4, 3]),   # bottom front left
        ]

        return Polyhedron3D(deformed, faces)

    @staticmethod
    def deformed_tetrahedron(vsk: vsketch.Vsketch, deform: float = 0.3) -> Polyhedron3D:
        """Create tetrahedron with random vertex deformation."""
        # Regular tetrahedron vertices
        vertices = [
            Vec3(1, 1, 1),
            Vec3(1, -1, -1),
            Vec3(-1, 1, -1),
            Vec3(-1, -1, 1),
        ]

        # Normalize to unit size and add deformation
        deformed = []
        for v in vertices:
            v_norm = v.normalize()
            deformed.append(Vec3(
                v_norm.x + vsk.random(-deform, deform),
                v_norm.y + vsk.random(-deform, deform),
                v_norm.z + vsk.random(-deform, deform)
            ))

        # 4 triangular faces (CCW winding)
        faces = [
            Face3D([0, 1, 2]),
            Face3D([0, 2, 3]),
            Face3D([0, 3, 1]),
            Face3D([1, 3, 2]),
        ]

        return Polyhedron3D(deformed, faces)

    @staticmethod
    def deformed_cube(vsk: vsketch.Vsketch, deform: float = 0.25) -> Polyhedron3D:
        """Create cube with random vertex deformation."""
        # Unit cube vertices
        vertices = [
            Vec3(-1, -1, -1),  # 0
            Vec3(1, -1, -1),   # 1
            Vec3(1, 1, -1),    # 2
            Vec3(-1, 1, -1),   # 3
            Vec3(-1, -1, 1),   # 4
            Vec3(1, -1, 1),    # 5
            Vec3(1, 1, 1),     # 6
            Vec3(-1, 1, 1),    # 7
        ]

        # Add deformation
        deformed = []
        for v in vertices:
            deformed.append(Vec3(
                v.x * 0.6 + vsk.random(-deform, deform),
                v.y * 0.6 + vsk.random(-deform, deform),
                v.z * 0.6 + vsk.random(-deform, deform)
            ))

        # 6 quad faces (CCW winding when viewed from outside)
        faces = [
            Face3D([0, 3, 2, 1]),  # back
            Face3D([4, 5, 6, 7]),  # front
            Face3D([0, 1, 5, 4]),  # bottom
            Face3D([2, 3, 7, 6]),  # top
            Face3D([0, 4, 7, 3]),  # left
            Face3D([1, 2, 6, 5]),  # right
        ]

        return Polyhedron3D(deformed, faces)


class StructureDrawer(ABC):
    """Base class for structure drawing with hatching support."""

    def __init__(self, vsk: vsketch.Vsketch, light_x: float, light_y: float, hatch_spacing: float = 0.12):
        self.vsk = vsk
        self.light_x = light_x
        self.light_y = light_y
        self.hatch_spacing = hatch_spacing

    @abstractmethod
    def draw(self, x: float, y: float, size: float, angle: float, detail: int) -> None:
        """Draw a structure at the given position."""
        pass

    def compute_light_direction(self, face_center_x: float, face_center_y: float) -> Vec3:
        """Compute 3D light direction from face center toward light source."""
        # 2D direction from face center to light source
        dx = self.light_x - face_center_x
        dy = self.light_y - face_center_y
        # Normalize in 2D, then add Y component (light from above)
        dist = math.sqrt(dx * dx + dy * dy) + 0.001
        return Vec3(dx / dist, 0.5, dy / dist).normalize()

    def compute_hatch_density(self, face_normal: Vec3, light_direction: Vec3) -> float:
        """Compute hatching density based on face orientation to light.

        Returns value 0.0 (brightest, no hatching) to 1.0 (darkest, dense hatching).
        """
        dot = face_normal.dot(light_direction)
        # Remap: facing light (dot=1) = no hatching (0), away (dot<=0) = dense (1)
        # Use squared falloff for more dramatic contrast
        brightness = max(0, dot)
        density = (1.0 - brightness) ** 0.7
        return density

    def draw_hatched_face(self, points: List[Tuple[float, float]], density: float,
                          face_center_x: float, face_center_y: float) -> None:
        """Draw a face with hatching at the given density."""
        if len(points) < 3:
            return

        # Calculate hatch angle perpendicular to axis from face center to light source
        angle_to_light = math.atan2(self.light_y - face_center_y, self.light_x - face_center_x)
        hatch_angle = angle_to_light + math.pi / 2

        # Calculate spacing inversely proportional to density
        # Very bright faces (low density) get very sparse or no hatching
        min_spacing = self.hatch_spacing * 0.5
        max_spacing = self.hatch_spacing * 4.0
        spacing = max_spacing - (max_spacing - min_spacing) * density

        # Get bounding box
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Diagonal for line length
        diagonal = math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Direction vectors
        dx = math.cos(hatch_angle)
        dy = math.sin(hatch_angle)
        # Perpendicular for stepping between lines
        px = -dy
        py = dx

        # Use layer 2 for the face outline (occluding polygon).
        # Layer 2 occludes layer 1 content that falls within the closed polygon.
        # This ensures proper occlusion: the face outline blocks plants/hatching behind it.
        self.vsk.stroke(2)
        self.vsk.polygon([p[0] for p in points], [p[1] for p in points], close=True)

        # Draw visible outline on layer 1 (so it's visible in the final output)
        self.vsk.stroke(1)
        self.vsk.polygon([p[0] for p in points], [p[1] for p in points], close=True)

        # Skip hatching for very bright faces (they appear as empty outlines)
        if density < 0.08:
            return

        # Generate and clip hatch lines
        num_lines = int(diagonal / spacing) + 2
        for i in range(-num_lines // 2, num_lines // 2 + 1):
            offset = i * spacing
            line_center_x = center_x + px * offset
            line_center_y = center_y + py * offset

            start_x = line_center_x - dx * diagonal
            start_y = line_center_y - dy * diagonal
            end_x = line_center_x + dx * diagonal
            end_y = line_center_y + dy * diagonal

            segments = self._clip_line_to_polygon(
                (start_x, start_y), (end_x, end_y), points
            )
            for seg_start, seg_end in segments:
                self.vsk.line(seg_start[0], seg_start[1], seg_end[0], seg_end[1])

    def _clip_line_to_polygon(self, p1: Tuple[float, float], p2: Tuple[float, float],
                               polygon: List[Tuple[float, float]]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Clip line segment to convex polygon using Cyrus-Beck algorithm."""
        t_enter = 0.0
        t_exit = 1.0

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        n = len(polygon)
        for i in range(n):
            # Edge from polygon[i] to polygon[(i+1) % n]
            edge_x = polygon[(i + 1) % n][0] - polygon[i][0]
            edge_y = polygon[(i + 1) % n][1] - polygon[i][1]

            # Inward normal (for CCW polygon, inward is to the right of edge)
            nx = edge_y
            ny = -edge_x

            # Vector from edge start to line start
            wx = p1[0] - polygon[i][0]
            wy = p1[1] - polygon[i][1]

            denom = nx * dx + ny * dy
            numer = -(nx * wx + ny * wy)

            if abs(denom) < 1e-10:
                # Line parallel to edge
                if numer < 0:
                    return []  # Outside
            else:
                t = numer / denom
                if denom < 0:
                    # Entering
                    t_enter = max(t_enter, t)
                else:
                    # Exiting
                    t_exit = min(t_exit, t)

        if t_enter > t_exit:
            return []  # No intersection

        clipped_start = (p1[0] + t_enter * dx, p1[1] + t_enter * dy)
        clipped_end = (p1[0] + t_exit * dx, p1[1] + t_exit * dy)

        return [(clipped_start, clipped_end)]


class PolyhedraDrawer(StructureDrawer):
    """Draws random polyhedra structures."""

    def draw(self, x: float, y: float, size: float, angle: float, detail: int) -> None:
        """Draw a polyhedron. detail controls shape complexity."""
        # Choose polyhedron type based on detail
        if detail <= 1:
            poly = PolyhedronGenerator.deformed_tetrahedron(self.vsk, 0.25)
        elif detail == 2:
            poly = PolyhedronGenerator.deformed_octahedron(self.vsk, 0.3)
        else:
            poly = PolyhedronGenerator.deformed_cube(self.vsk, 0.2)

        # Apply rotations
        poly = poly.rotate_y(angle)
        poly = poly.rotate_x(self.vsk.random(-0.5, 0.5))

        # Project to 2D
        projected_faces = poly.project_to_2d(size * 0.5, x, y)

        # Draw faces back-to-front (outlines and hatching)
        for face in projected_faces:
            # Compute face center in 2D
            face_center_x = sum(p[0] for p in face.points_2d) / len(face.points_2d)
            face_center_y = sum(p[1] for p in face.points_2d) / len(face.points_2d)

            # Compute light direction from this face's position
            light_dir = self.compute_light_direction(face_center_x, face_center_y)
            density = self.compute_hatch_density(face.normal_3d, light_dir)
            self.draw_hatched_face(face.points_2d, density, face_center_x, face_center_y)

        # Face outlines are drawn on layer 2 (for occlusion) and layer 1 (visible outline).
        # Hatching is on layer 1. Layer 2 occludes layer 1, ensuring proper depth sorting.
