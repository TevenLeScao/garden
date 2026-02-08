"""Claude Garden - A generative garden sketch with occlusion."""

import math
import vsketch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union
from plants import PlantDrawer, GrassDrawer, BerryPlantDrawer, LeafyPlantDrawer, BranchDrawer
from structures import StructureDrawer, PolyhedraDrawer


@dataclass
class Plant:
    """A plant instance to be drawn."""
    x: float
    y: float
    size: float
    angle: float
    drawer_name: str
    draw_order: float = field(default=0.0)


@dataclass
class Structure:
    """A structure instance to be drawn."""
    x: float
    y: float
    size: float
    angle: float
    drawer_name: str
    draw_order: float = field(default=0.0)


# Registry of available plant types with their weights and drawer classes
PLANT_REGISTRY: Dict[str, Dict] = {
    "grass": {
        "drawer_class": GrassDrawer,
        "weight": 55,
        "size_scale": 0.3,  # Grass is much smaller than other plants
        "draw_params": lambda vsk: {"detail": int(vsk.random(6, 16))},
    },
    "leafy": {
        "drawer_class": LeafyPlantDrawer,
        "weight": 18,
        "size_scale": 1.0,
        "draw_params": lambda vsk: {"detail": int(vsk.random(4, 10))},
    },
    "berry": {
        "drawer_class": BerryPlantDrawer,
        "weight": 15,
        "size_scale": 1.0,
        "draw_params": lambda vsk: {"detail": int(vsk.random(3, 5))},
    },
    "branch": {
        "drawer_class": BranchDrawer,
        "weight": 12,
        "size_scale": 1.0,
        "draw_params": lambda vsk: {"detail": int(vsk.random(4, 6))},
    },
}

# Registry of available structure types
STRUCTURE_REGISTRY: Dict[str, Dict] = {
    "polyhedra": {
        "drawer_class": PolyhedraDrawer,
        "weight": 100,
        "size_scale": 1.0,
        "draw_params": lambda vsk: {"detail": int(vsk.random(1, 4))},
    },
}


class ClaudeGardenSketch(vsketch.SketchClass):
    # === Canvas ===
    page_size = vsketch.Param("a4", choices=["a4", "a3", "letter", "11x14in"])
    landscape = vsketch.Param(True)
    margin = vsketch.Param(1.5, min_value=0.0, max_value=5.0, step=0.1)

    # === Garden ===
    density = vsketch.Param(15, min_value=5, max_value=50)
    plant_size = vsketch.Param(5.0, min_value=1.0, max_value=15.0, step=0.5)
    size_variance = vsketch.Param(0.6, min_value=0.0, max_value=1.0, step=0.05)
    wind = vsketch.Param(0.2, min_value=-1.0, max_value=1.0, step=0.05)

    # === Structures ===
    structure_count = vsketch.Param(10, min_value=0, max_value=30)
    structure_size = vsketch.Param(6.0, min_value=1.0, max_value=15.0, step=0.5)
    structure_size_variance = vsketch.Param(0.7, min_value=0.0, max_value=1.0, step=0.05)
    light_angle = vsketch.Param(0.5, min_value=-3.14, max_value=3.14, step=0.1)
    hatch_spacing = vsketch.Param(0.25, min_value=0.1, max_value=0.5, step=0.02)

    # === Style ===
    use_occlusion = vsketch.Param(True)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size(self.page_size, landscape=self.landscape)
        vsk.scale("cm")

        width, height = self._get_drawing_area(vsk)

        # Create all drawers
        plant_drawers = self._create_plant_drawers(vsk)
        structure_drawers = self._create_structure_drawers(vsk)

        # Generate all drawable objects
        plants = self._generate_plants(vsk, width, height)
        structures = self._generate_structures(vsk, width, height)

        # Combine into unified draw list: (draw_order, type, object)
        draw_list: List[Tuple[float, str, Union[Plant, Structure]]] = []
        for plant in plants:
            draw_list.append((plant.draw_order, 'plant', plant))
        for structure in structures:
            draw_list.append((structure.draw_order, 'structure', structure))

        # Sort by draw order (lower = drawn first = behind)
        draw_list.sort(key=lambda x: x[0])

        # Draw everything in order
        for draw_order, obj_type, obj in draw_list:
            if obj_type == 'plant':
                drawer = plant_drawers[obj.drawer_name]
                config = PLANT_REGISTRY[obj.drawer_name]
                params = config["draw_params"](vsk)
                size_scale = config.get("size_scale", 1.0)
                drawer.draw(obj.x, obj.y, obj.size * size_scale, obj.angle, params["detail"])
            else:
                drawer = structure_drawers[obj.drawer_name]
                config = STRUCTURE_REGISTRY[obj.drawer_name]
                params = config["draw_params"](vsk)
                size_scale = config.get("size_scale", 1.0)
                drawer.draw(obj.x, obj.y, obj.size * size_scale, obj.angle, params["detail"])

    def _get_drawing_area(self, vsk: vsketch.Vsketch) -> tuple:
        """Calculate drawing area dimensions and set up coordinate system."""
        px_per_cm = 96 / 2.54
        full_width = vsk.width / px_per_cm
        full_height = vsk.height / px_per_cm

        width = full_width - 2 * self.margin
        height = full_height - 2 * self.margin

        # Translate from center to top-left corner with margin
        vsk.translate(-full_width / 2 + self.margin, -full_height / 2 + self.margin)

        return width, height

    def _create_plant_drawers(self, vsk: vsketch.Vsketch) -> Dict[str, PlantDrawer]:
        """Create drawer instances for all registered plant types."""
        drawers = {}
        for name, config in PLANT_REGISTRY.items():
            drawer_class = config["drawer_class"]
            if drawer_class == GrassDrawer:
                grass_curve = 0.5 + self.wind * 0.3
                drawers[name] = drawer_class(vsk, self.use_occlusion, 0.05, grass_curve, self.wind)
            else:
                drawers[name] = drawer_class(vsk, self.use_occlusion, 0.03, self.wind)
        return drawers

    def _create_structure_drawers(self, vsk: vsketch.Vsketch) -> Dict[str, StructureDrawer]:
        """Create drawer instances for all registered structure types."""
        drawers = {}
        for name, config in STRUCTURE_REGISTRY.items():
            drawer_class = config["drawer_class"]
            drawers[name] = drawer_class(vsk, self.use_occlusion, self.light_angle, self.hatch_spacing)
        return drawers

    def _generate_plants(self, vsk: vsketch.Vsketch, width: float, height: float) -> List[Plant]:
        """Generate plant instances with random positions and layer ordering."""
        plants = []
        total_plants = int(self.density * width / 1.5)

        for _ in range(total_plants):
            x = vsk.random(0, width)
            y = vsk.random(0, height)

            # Draw order: random with slight bias toward front (higher y = drawn later)
            # y_bias ranges 0-1 based on vertical position
            y_bias = y / height
            draw_order = vsk.random(0, 1) + y_bias * 0.5

            angle = (x - width / 2) / width * 0.3 + self.wind * 0.5
            size = self._random_size(vsk)
            plant_type = self._random_plant_type(vsk)

            plants.append(Plant(x, y, size, angle, plant_type, draw_order))

        return plants

    def _random_plant_type(self, vsk: vsketch.Vsketch) -> str:
        """Select plant type based on registry weights."""
        total_weight = sum(config["weight"] for config in PLANT_REGISTRY.values())
        r = vsk.random(0, total_weight)

        cumulative = 0
        for name, config in PLANT_REGISTRY.items():
            cumulative += config["weight"]
            if r < cumulative:
                return name

        return list(PLANT_REGISTRY.keys())[-1]

    def _random_size(self, vsk: vsketch.Vsketch) -> float:
        """Get random size with variance."""
        min_size = self.plant_size * (1 - self.size_variance)
        max_size = self.plant_size * (1 + self.size_variance)
        return vsk.random(min_size, max_size)

    def _generate_structures(self, vsk: vsketch.Vsketch, width: float, height: float) -> List[Structure]:
        """Generate structure instances biased toward being behind plants, without overlapping."""
        structures = []
        max_attempts = self.structure_count * 10

        for _ in range(max_attempts):
            if len(structures) >= self.structure_count:
                break

            x = vsk.random(0, width)
            y = vsk.random(0, height)
            size = self._random_structure_size(vsk)

            # Check for overlap with existing structures
            overlaps = False
            for existing in structures:
                dist = math.sqrt((x - existing.x) ** 2 + (y - existing.y) ** 2)
                min_dist = (size + existing.size) * 0.6  # 0.6 factor for some spacing
                if dist < min_dist:
                    overlaps = True
                    break

            if overlaps:
                continue

            # Draw order biased toward LOWER values (drawn first = behind)
            y_bias = y / height
            draw_order = vsk.random(-0.3, 0.7) + y_bias * 0.3

            angle = vsk.random(0, 2 * math.pi)

            structures.append(Structure(x, y, size, angle, "polyhedra", draw_order))

        return structures

    def _random_structure_size(self, vsk: vsketch.Vsketch) -> float:
        """Get random size with distribution skewed toward smaller sizes."""
        # Power distribution: smaller values more likely
        t = vsk.random(0, 1) ** 2.5

        min_size = self.structure_size * (1 - self.structure_size_variance)
        max_size = self.structure_size * (1 + self.structure_size_variance)

        return min_size + t * (max_size - min_size)

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        # Crop to page bounds, apply occlusion, and optimize paths
        crop_cmd = f"crop 0 0 {vsk.width}px {vsk.height}px"
        # Use occult -i to perform occlusion ignoring layers (treats all geometry as one layer)
        # Later-drawn shapes occlude earlier-drawn ones (painter's algorithm)
        occlusion_cmd = "occult -i" if self.use_occlusion else ""
        vsk.vpype(f"{crop_cmd} {occlusion_cmd} linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    ClaudeGardenSketch.display()
