"""Plant drawing classes for the Claude Garden sketch.

To add a new plant type:
1. Create a new class inheriting from PlantDrawer
2. Implement the draw(x, y, size, angle, detail) method
3. Register it in PLANT_REGISTRY in sketch_claude_garden.py
"""

import math
from abc import ABC, abstractmethod
from typing import Tuple, List
import vsketch


def bezier_point(t: float, p0: Tuple[float, float], p1: Tuple[float, float],
                 p2: Tuple[float, float], p3: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate a point on a cubic bezier curve."""
    t1 = 1 - t
    x = t1**3 * p0[0] + 3*t1**2*t * p1[0] + 3*t1*t**2 * p2[0] + t**3 * p3[0]
    y = t1**3 * p0[1] + 3*t1**2*t * p1[1] + 3*t1*t**2 * p2[1] + t**3 * p3[1]
    return x, y


class PlantDrawer(ABC):
    """Base class for plant drawing.

    Subclasses must implement the draw() method. Use the helper methods
    for drawing shapes:
    - draw_occluding_shape_from_curves(): For leaf/blade shapes
    - draw_occluding_circle(): For berries/flower centers
    - draw_occluding_polygon(): For custom polygon shapes
    """

    def __init__(self, vsk: vsketch.Vsketch, wind: float = 0.0):
        self.vsk = vsk
        self.wind = wind

    @abstractmethod
    def draw(self, x: float, y: float, size: float, angle: float, detail: int) -> None:
        """Draw a plant at the given position.

        Args:
            x: Base x position
            y: Base y position (ground level)
            size: Overall size multiplier
            angle: Lean angle (affected by position and wind)
            detail: Level of detail (blade count, branch depth, etc.)
        """
        pass

    def draw_occluding_polygon(self, points_x: List[float], points_y: List[float]) -> None:
        """Draw a closed polygon."""
        self.vsk.stroke(1)
        self.vsk.polygon(points_x, points_y, close=True)

    def draw_occluding_shape_from_curves(self, left_curve: List[Tuple[float, float]],
                                          right_curve: List[Tuple[float, float]]) -> None:
        """Draw a shape defined by two curves meeting at endpoints as a closed polygon.

        For proper occlusion, we need a closed polygon that vpype-occult can recognize.
        We combine both curves into a single closed path.
        """
        self.vsk.stroke(1)
        # Create a closed polygon from both curves
        # left_curve goes from base to tip, right_curve also goes from base to tip
        # We need: left_curve (base->tip) + reversed right_curve (tip->base) = closed shape
        all_points = list(left_curve) + list(reversed(right_curve))
        if len(all_points) >= 3:
            self.vsk.polygon([p[0] for p in all_points], [p[1] for p in all_points], close=True)

    def draw_occluding_circle(self, cx: float, cy: float, radius: float) -> None:
        """Draw a circle."""
        self.vsk.stroke(1)
        self.vsk.circle(cx, cy, radius, mode="radius")


class GrassDrawer(PlantDrawer):
    """Draws grass blade clumps."""

    def __init__(self, vsk: vsketch.Vsketch, blade_width: float, grass_curve: float, wind: float = 0.0):
        super().__init__(vsk, wind)
        self.blade_width = blade_width
        self.grass_curve = grass_curve

    def draw(self, x: float, y: float, size: float, angle: float, detail: int) -> None:
        """Draw a clump of grass blades. detail = base blade count, scaled by size."""
        # Scale blade count with size - smaller grass gets fewer blades
        # Reference size ~1.5 (typical grass size with size_scale=0.3 and plant_size=5)
        size_factor = max(0.3, size / 1.5)
        blade_count = max(2, int(detail * size_factor))
        for _ in range(blade_count):
            blade_x = x + self.vsk.random(-0.3, 0.3) * size * 0.3
            blade_height = size * self.vsk.random(0.6, 1.0)
            curve_dir = self.vsk.random(-1, 1) + angle * 2 + self.wind * 1.5
            curve_amount = self.grass_curve * self.vsk.random(0.5, 1.5)

            self._draw_blade(blade_x, y, blade_height, curve_dir, curve_amount)

    def _draw_blade(self, x: float, y: float, height: float,
                    curve_dir: float, curve_amount: float) -> None:
        """Draw a single grass blade as two curves meeting at a sharp tip."""
        tip_x = x + curve_dir * height * curve_amount
        tip_y = y - height
        base_half_width = self.blade_width * height / 2

        p0 = (x, y)
        p1 = (x + curve_dir * height * curve_amount * 0.2, y - height * 0.4)
        p2 = (x + curve_dir * height * curve_amount * 0.7, y - height * 0.8)
        p3 = (tip_x, tip_y)

        steps = 8
        left_curve = []
        right_curve = []

        for i in range(steps + 1):
            t = i / steps
            cx, cy = bezier_point(t, p0, p1, p2, p3)
            width = base_half_width * (1 - t ** 0.7)

            if i < steps:
                cx_next, cy_next = bezier_point((i + 1) / steps, p0, p1, p2, p3)
                dx, dy = cx_next - cx, cy_next - cy
            else:
                dx, dy = tip_x - cx, tip_y - cy

            length = math.sqrt(dx*dx + dy*dy) + 0.001
            nx, ny = -dy/length, dx/length

            left_curve.append((cx + nx * width, cy + ny * width))
            right_curve.append((cx - nx * width, cy - ny * width))

        self.draw_occluding_shape_from_curves(left_curve, right_curve)


class BerryPlantDrawer(PlantDrawer):
    """Draws branching berry plants."""

    def __init__(self, vsk: vsketch.Vsketch, stem_width: float, wind: float = 0.0):
        super().__init__(vsk, wind)
        self.stem_width = stem_width

    def draw(self, x: float, y: float, size: float, angle: float, detail: int) -> None:
        """Draw a berry plant. detail = branch recursion depth."""
        berry_size = size * 0.025
        self._draw_branch(x, y, -math.pi / 2 + angle * 0.3 + self.wind * 0.3,
                         size * 0.35, detail, self.stem_width * size, berry_size)

    def _draw_branch(self, x: float, y: float, angle: float, length: float,
                     depth: int, width: float, berry_size: float) -> None:
        if depth <= 0:
            return

        angle += self.wind * 0.15

        end_x = x + math.cos(angle) * length
        end_y = y + math.sin(angle) * length

        self._draw_tapered_branch(x, y, end_x, end_y, width, width * 0.6)

        if depth == 1:
            self.draw_occluding_circle(end_x, end_y, berry_size)
        else:
            num_branches = int(self.vsk.random(2, 4))
            angle_spread = math.radians(30)

            for i in range(num_branches):
                branch_angle = angle + self.vsk.random(-angle_spread, angle_spread)
                branch_length = length * self.vsk.random(0.6, 0.9)
                self._draw_branch(end_x, end_y, branch_angle,
                                 branch_length, depth - 1, width * 0.7, berry_size)

    def _draw_tapered_branch(self, x1: float, y1: float, x2: float, y2: float,
                             width_start: float, width_end: float) -> None:
        """Draw a tapered branch segment."""
        steps = 4
        left_curve = []
        right_curve = []

        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx*dx + dy*dy) + 0.001
        nx, ny = -dy/length, dx/length

        for i in range(steps + 1):
            t = i / steps
            px = x1 + dx * t
            py = y1 + dy * t
            w = (width_start * (1-t) + width_end * t) / 2

            left_curve.append((px + nx * w, py + ny * w))
            right_curve.append((px - nx * w, py - ny * w))

        self.draw_occluding_shape_from_curves(left_curve, right_curve)


class LeafyPlantDrawer(PlantDrawer):
    """Draws plants with serrated leaves."""

    def __init__(self, vsk: vsketch.Vsketch, stem_width: float, wind: float = 0.0):
        super().__init__(vsk, wind)
        self.stem_width = stem_width

    def draw(self, x: float, y: float, size: float, angle: float, detail: int) -> None:
        """Draw a leafy plant. detail = leaf count."""
        stem_height = size * 0.8
        stem_curve_factor = self.vsk.random(-0.15, 0.15) + angle * 0.3 + self.wind * 0.3
        stem_top_x = x + stem_curve_factor * size * 0.4
        stem_top_y = y - stem_height

        # Compute curve control point for stem (same logic as _draw_curved_stem)
        dx, dy = stem_top_x - x, stem_top_y - y
        length = math.sqrt(dx*dx + dy*dy) + 0.001
        curve_amount = self.vsk.random(-0.3, 0.3) * length + self.wind * length * 0.2
        mid_x = (x + stem_top_x) / 2 + (-dy / length) * curve_amount
        mid_y = (y + stem_top_y) / 2 + (dx / length) * curve_amount

        # Draw the curved stem (uses same random state, so we reseed for consistency)
        self._draw_curved_stem_with_control(x, y, stem_top_x, stem_top_y, mid_x, mid_y, self.stem_width * size)

        for i in range(detail):
            t = (i + 1) / (detail + 1)
            # Place leaves along the curved stem (quadratic bezier)
            t1 = 1 - t
            leaf_x = t1**2 * x + 2*t1*t * mid_x + t**2 * stem_top_x
            leaf_y = t1**2 * y + 2*t1*t * mid_y + t**2 * stem_top_y

            side = 1 if i % 2 == 0 else -1
            leaf_size = size * 0.25 * (1 - t * 0.4)
            # Base angle: right-pointing leaves ~50°, left-pointing leaves ~130° (π - 50°)
            base_leaf_angle = math.radians(50 + self.vsk.random(-10, 10))
            if side == -1:
                base_leaf_angle = math.pi - base_leaf_angle  # Mirror to point left
            leaf_angle = base_leaf_angle + angle * 0.2 + self.wind * 0.2

            self._draw_serrated_leaf(leaf_x, leaf_y, leaf_size, leaf_angle)

    def _draw_curved_stem_with_control(self, x1: float, y1: float, x2: float, y2: float,
                                        mid_x: float, mid_y: float, width: float) -> None:
        """Draw stem as a curved tapered shape using a quadratic bezier control point."""
        steps = 8
        left_curve = []
        right_curve = []

        for i in range(steps + 1):
            t = i / steps
            t1 = 1 - t
            # Quadratic bezier curve through control point
            px = t1**2 * x1 + 2*t1*t * mid_x + t**2 * x2
            py = t1**2 * y1 + 2*t1*t * mid_y + t**2 * y2

            # Compute tangent direction for perpendicular offset
            if i < steps:
                t_next = (i + 1) / steps
                t1_next = 1 - t_next
                px_next = t1_next**2 * x1 + 2*t1_next*t_next * mid_x + t_next**2 * x2
                py_next = t1_next**2 * y1 + 2*t1_next*t_next * mid_y + t_next**2 * y2
                tdx, tdy = px_next - px, py_next - py
            else:
                tdx, tdy = x2 - mid_x, y2 - mid_y

            tlen = math.sqrt(tdx*tdx + tdy*tdy) + 0.001
            nx, ny = -tdy/tlen, tdx/tlen

            w = width * (1 - t * 0.4) / 2

            left_curve.append((px + nx * w, py + ny * w))
            right_curve.append((px - nx * w, py - ny * w))

        self.draw_occluding_shape_from_curves(left_curve, right_curve)

    def _draw_serrated_leaf(self, x: float, y: float, size: float, angle: float) -> None:
        """Draw a serrated leaf."""
        leaf_length = size
        leaf_width = size * 0.35
        serration_count = 4
        serration_depth = leaf_width * 0.15

        left_curve = [(x, y)]
        right_curve = [(x, y)]

        for i in range(serration_count):
            t = (i + 0.5) / serration_count
            cx = x + math.cos(angle) * leaf_length * t
            cy = y + math.sin(angle) * leaf_length * t
            width_factor = 1 - abs(t - 0.5) * 2
            local_width = leaf_width * width_factor

            perp_left = angle + math.pi / 2
            perp_right = angle - math.pi / 2

            left_curve.append((cx + math.cos(perp_left) * (local_width + serration_depth),
                              cy + math.sin(perp_left) * (local_width + serration_depth)))
            right_curve.append((cx + math.cos(perp_right) * (local_width + serration_depth),
                               cy + math.sin(perp_right) * (local_width + serration_depth)))

            if i < serration_count - 1:
                t2 = (i + 1) / serration_count
                cx2 = x + math.cos(angle) * leaf_length * t2
                cy2 = y + math.sin(angle) * leaf_length * t2
                width_factor2 = 1 - abs(t2 - 0.5) * 2
                local_width2 = leaf_width * width_factor2

                left_curve.append((cx2 + math.cos(perp_left) * local_width2,
                                  cy2 + math.sin(perp_left) * local_width2))
                right_curve.append((cx2 + math.cos(perp_right) * local_width2,
                                   cy2 + math.sin(perp_right) * local_width2))

        tip_x = x + math.cos(angle) * leaf_length
        tip_y = y + math.sin(angle) * leaf_length
        left_curve.append((tip_x, tip_y))
        right_curve.append((tip_x, tip_y))

        # Don't reverse right_curve here - draw_occluding_shape_from_curves handles it
        self.draw_occluding_shape_from_curves(left_curve, right_curve)

        self.vsk.stroke(1)
        self.vsk.line(x, y, tip_x, tip_y)


class BranchDrawer(PlantDrawer):
    """Draws bare tree branches."""

    def __init__(self, vsk: vsketch.Vsketch, stem_width: float, wind: float = 0.0):
        super().__init__(vsk, wind)
        self.stem_width = stem_width

    def draw(self, x: float, y: float, size: float, angle: float, detail: int) -> None:
        """Draw tree branches. detail = recursion depth."""
        self._draw_segment(x, y, -math.pi / 2 + angle * 0.2 + self.wind * 0.2,
                          size * 0.3, detail, self.stem_width * size * 1.5)

    def _draw_segment(self, x: float, y: float, angle: float, length: float,
                      depth: int, width: float) -> None:
        if depth <= 0 or length < 0.08:
            return

        angle += self.wind * 0.1

        curve_offset = self.vsk.random(-0.15, 0.15) * length + self.wind * length * 0.1
        end_x = x + math.cos(angle) * length
        end_y = y + math.sin(angle) * length

        # Terminal branches (depth == 1) should taper to a closed point
        # Non-terminal branches keep some width for child branches to connect
        width_end = 0.0 if depth == 1 else width * 0.5
        self._draw_curved_branch(x, y, end_x, end_y, width, width_end, curve_offset, angle)

        if depth > 1:
            num_branches = int(self.vsk.random(2, 4))
            angle_spread = math.radians(30)

            for i in range(num_branches):
                if num_branches == 2:
                    branch_angle = angle + ((-1) ** i) * self.vsk.random(angle_spread * 0.5, angle_spread)
                else:
                    offset = (i - (num_branches - 1) / 2) / max(1, num_branches - 1) * 2
                    branch_angle = angle + offset * angle_spread + self.vsk.random(-0.15, 0.15)

                branch_length = length * self.vsk.random(0.5, 0.75)
                self._draw_segment(end_x, end_y, branch_angle,
                                  branch_length, depth - 1, width * 0.6)

    def _draw_curved_branch(self, x1: float, y1: float, x2: float, y2: float,
                            width_start: float, width_end: float,
                            curve_offset: float, angle: float) -> None:
        """Draw a curved tapered branch."""
        steps = 5
        left_curve = []
        right_curve = []

        perp = angle + math.pi / 2
        mid_x = (x1 + x2) / 2 + math.cos(perp) * curve_offset
        mid_y = (y1 + y2) / 2 + math.sin(perp) * curve_offset

        for i in range(steps + 1):
            t = i / steps
            t1 = 1 - t
            px = t1**2 * x1 + 2*t1*t * mid_x + t**2 * x2
            py = t1**2 * y1 + 2*t1*t * mid_y + t**2 * y2

            w = (width_start * (1-t) + width_end * t) / 2

            if i < steps:
                t_next = (i + 1) / steps
                t1_next = 1 - t_next
                px_next = t1_next**2 * x1 + 2*t1_next*t_next * mid_x + t_next**2 * x2
                py_next = t1_next**2 * y1 + 2*t1_next*t_next * mid_y + t_next**2 * y2
                dx, dy = px_next - px, py_next - py
            else:
                dx, dy = x2 - mid_x, y2 - mid_y

            length = math.sqrt(dx*dx + dy*dy) + 0.001
            nx, ny = -dy/length, dx/length

            left_curve.append((px + nx * w, py + ny * w))
            right_curve.append((px - nx * w, py - ny * w))

        self.draw_occluding_shape_from_curves(left_curve, right_curve)
