import numpy as np
import pyglet
from typing import Tuple

from gym.envs.classic_control.rendering import Viewer as GymViewer, Geom, Line

# xxx(okachaiev): setup proper color pallet structure
green = (0,158,115)
yellow = (240,228,66)
blue = (0,114,178)
deepskyblue = (0,191,255)
pink = (204,121,167)
orange = (213,94,0)
silver = (192,192,192)
slategray = (112,128,144)
darkred = (114,0,0)

# xxx(okachaiev): make proper unit configuration
unit_config = {
    "Worker": (silver, 0.5),
    "Light": (yellow, 0.6),
    "Heavy": (orange, 0.6),
    "Ranged": (deepskyblue, 0.8),
}

building_config = {
    "Base": (silver,),
    "Barracks": (slategray,)
}

direction_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0),]

# xxx(okachaiev): would be nice to have geometry over point
# with overloaded + and *
def translate_direction(direction, length):
    offset_x, offset_y = direction_offsets[direction]
    return (offset_x*length, offset_y*length)

# xxx(okachaiev): same here, no need for indications
class Proxy(Geom):

    def __init__(self, obj):
        self._obj = obj
    
    def render(self):
        self._obj.draw()


# xxx(okachaiev): extend to all contstants
class ActionType:
    PRODUCE = 4


# xxx(okachaiev): I should definitely separate the concept of
# "screen" and "renderer engine" to avoid mixing the logic for
# dealing with GL vertices together with the logic of the game
class Viewer:

    def __init__(self, height:int=640, width:int=640, title:str="MicroRTS"):
        self._height = height
        self._width = width
        # xxx(okachaiev): i actually don't need this indirection
        self._viewer = GymViewer(height, width)
        self._viewer.window.set_caption(title)
        self._init_grid()
        self._reset_canvas()
        # xxx(okachaiev): add text label with bot names and stats

    def _init_grid(self):
        self._map_height, self._map_width = 16, 16
        self._offset = 20
        xs, self._step = np.linspace(
            self._offset, self._width-self._offset, self._map_width+1, retstep=True
        )
        # xxx(okachaiev): this won't work for non-squared maps
        # i need to fit squares into min dimention and center to fit max dimension
        self._centers = xs[:-1]+self._step/2
        for start in xs:
            # xxx(okachaiev): do not use Gym shapes, pyglet is fine as is
            # though I need to remember to keep it off of canvas list
            # so they don't disappear on "reset"
            self._viewer.add_geom(Line((start, self._offset), (start, self._width-self._offset)))
            self._viewer.add_geom(Line((self._offset, start), (self._width-self._offset, start)))

    def _init_layers(self):
        self._batch = pyglet.graphics.Batch()
        self._viewer.add_geom(Proxy(self._batch))
        self._groups = {
            "background": pyglet.graphics.OrderedGroup(0),
            "circle_foreground": pyglet.graphics.OrderedGroup(1),
            "texts": pyglet.graphics.OrderedGroup(2),
            "progress": pyglet.graphics.OrderedGroup(3),
        }

    # xxx(okachaiev): it might be not the best approach to redefine
    # all geometries but it's a good enough starting point. later
    # i can reimplement it to track changes in already defined shapes
    def _reset_canvas(self):
        if hasattr(self, "_canvas"):
            for v in self._canvas:
                if hasattr(v, "delete"):
                    v.delete()
            for label in self._labels:
                label.delete()
        self._canvas = []
        self._labels = []
        # xxx(okachaiev): dropping entire batch is brutal
        self._init_layers()

    def _add_to_canvas(self, *geoms):
        for geom in geoms:
            self._canvas.append(geom)

    def _add_label_to_canvas(self, *labels):
        for label in labels:
            self._labels.append(label)

    def _cell_to_coords(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        row, col = cell
        return self._centers[col-1], self._width-self._centers[row-1]

    def _add_resource_label_geom(self, cell_coords, resources, font_size=12):
        if resources == 0: return None
        x, y = cell_coords
        geom = pyglet.text.Label(
            str(resources),
            font_size=font_size,
            color=(0,0,0,255),
            bold=True,
            x=x, y=y,
            anchor_x="center", anchor_y="center",
            batch=self._batch, group=self._groups["texts"]
        )
        self._add_label_to_canvas(geom)
        return geom

    def _add_resource_geom(self, cell, resources):
        if resources == 0: return None
        x, y = self._cell_to_coords(cell)
        rect = pyglet.shapes.Rectangle(
            x-self._step/2, y-self._step/2, self._step, self._step,
            color=green,
            batch=self._batch, group=self._groups["background"]
        )
        self._add_to_canvas(rect)
        text = self._add_resource_label_geom((x, y), resources)
        return (rect, text)

    # xxx(okachaiev): process hp/max_hp progress
    def _add_building_geom(self, cell, building_type, hp, max_hp, resources, owner):
        x, y = self._cell_to_coords(cell)
        color, = building_config[building_type]
        border_color = blue if owner == 1 else pink
        rect = pyglet.shapes.BorderedRectangle(
            x-self._step/2, y-self._step/2, self._step, self._step,
            border=2,
            color=color, border_color=border_color,
            batch=self._batch, group=self._groups["background"]
        )
        self._add_to_canvas(rect)
        text = self._add_resource_label_geom((x, y), resources)

        return rect, text

    def _add_unit_tick_geom(self, cell_coors, radius, direction, color):
        if direction is None: return None
        if direction > 3: return None
        x, y = cell_coors
        offset_x, offset_y = translate_direction(direction, radius)
        offset_x_hat, offset_y_hat = translate_direction(direction, self._step)
        line = pyglet.shapes.Line(
            x+offset_x, y+offset_y, x+offset_x_hat, y+offset_y_hat,
            width=2, color=color,
            batch=self._batch, group=self._groups["texts"]
        )
        self._add_to_canvas(line)
        return line

    # xxx(okachaiev): "direction" should be probably splited into "action"
    # or something (as it might mean different things)
    def _add_unit_geom(self, cell, unit_type, hp, max_hp, direction, resources, owner):
        x, y = self._cell_to_coords(cell)
        color, radius = unit_config[unit_type]
        border_color = blue if owner == 1 else pink
        back_circle = pyglet.shapes.Circle(
            x, y, radius*self._step/2+2,
            color=border_color,
            batch=self._batch, group=self._groups["background"]
        )
        circle = pyglet.shapes.Circle(
            x, y, radius*self._step/2,
            color=color,
            batch=self._batch, group=self._groups["circle_foreground"]
        )
        self._add_to_canvas(back_circle, circle)
        text = self._add_resource_label_geom((x, y), resources, font_size=9)
        tick = self._add_unit_tick_geom((x,y), radius*self._step/2, direction, border_color)
        hp_progress = None
        if hp < max_hp:
            # xxx(okachaiev): progress bar for units could look like missing segment
            # rather than the bar on top of them
            hp_progress = self._add_progress_bar_geom((x,y), hp/max_hp, color, darkred)
        return back_circle, circle, text, tick, hp_progress

    def _add_progress_bar_geom(self, cell_coords, progress, color, background_color):
        x, y = cell_coords
        left_bar = pyglet.shapes.Rectangle(
            x-self._step/2, y+self._step/2-(0.2*self._step), self._step*progress, 0.2*self._step,
            color=color,
            batch=self._batch, group=self._groups["progress"]
        )
        self._add_to_canvas(left_bar)
        right_bar = None
        if background_color is not None:
            right_bar = pyglet.shapes.Rectangle(
                x-self._step/2+self._step*progress, y+self._step/2-0.2*self._step,
                self._step*(1-progress), 0.2*self._step,
                color=background_color,
                batch=self._batch, group=self._groups["progress"]
            )
            self._add_to_canvas(right_bar)
        return left_bar, right_bar

    def _add_production_geom(self, cell, progress, label_text, owner):
        x, y = self._cell_to_coords(cell)
        color = blue if owner == 1 else pink
        bar = self._add_progress_bar_geom((x,y), progress, color, None)
        text = pyglet.text.Label(
            label_text,
            font_size=8,
            color=color + (255,),
            bold=True,
            x=x, y=y,
            anchor_x="center", anchor_y="center",
            batch=self._batch, group=self._groups["texts"]
        )
        self._add_label_to_canvas(text)
        return bar, text

    def render(self, gs):
        self._reset_canvas()

        for unit in gs.getUnits():
            action = gs.getActionAssignment(unit)
            cell = (unit.getY()+1, unit.getX()+1)
            if unit.getType().isResource:
                self._add_resource_geom(cell, unit.getResources())
            elif unit.getType().canMove:
                self._add_unit_geom(
                    cell,
                    unit.getType().name,
                    unit.getHitPoints(),
                    unit.getMaxHitPoints(),
                    None if not action else action.action.getDirection(),
                    unit.getResources(),
                    unit.getPlayer()+1, # xxx(kachaiev): should it be +1?
                )
            elif unit.getType().name in building_config:
                # should be a building
                self._add_building_geom(
                    cell,
                    unit.getType().name,
                    unit.getHitPoints(),
                    unit.getMaxHitPoints(),
                    unit.getResources(),
                    unit.getPlayer()+1, # xxx(kachaiev): should it be +1?
                )
                if action is not None and action.action.getType() == ActionType.PRODUCE:
                    eta = action.time + action.action.ETA(action.unit) - gs.getTime()
                    progress = (1 - (eta / action.action.ETA(action.unit)))
                    label = str(action.action.getUnitType().name)
                    direction = action.action.getDirection()
                    offset_x, offset_y = direction_offsets[direction]
                    action_cell = (unit.getY()+1+offset_y, unit.getX()+1+offset_x)
                    self._add_production_geom(action_cell, progress, label, unit.getPlayer()+1)

        self._viewer.render()

    def tick(self):
        """Dispatch event without re-rendering"""
        self._viewer.window.dispatch_events()

    def close(self):
        if self._viewer:
            self._reset_canvas()
            self._viewer.close()
        self._viewer = None

    def __del__(self):
        self.close()


# g1 = resource_geom((1,1), 25)
# g2 = resource_geom((2,1), 20)
# g3 = resource_geom((15,16), 20)
# g4 = resource_geom((16,16), 25)

# g5 = building_geom((3,3), "base", 3, None, 3, 1)
# g6 = building_geom((14,14), "base", 3, None, 7, 2)
# g7 = building_geom((3,7), "barracks", 3, None, 0, 1)

# g8 = unit_geom((3,8), "worker", 20, None, 1, 1)
# g9 = unit_geom((8,3), "light", 20, "w", 0, 2)
# g10 = unit_geom((12,12), "heavy", 3, None, 3, 2)
# g11 = unit_geom((10,14), "range", 5, "n", 0, 2)

# g12 = production_geom((9,9), 0.5, "worker", 1)
# g13 = production_geom((11,9), 0.8, "worker", 2)
