import numpy as np
import sys
from typing import List, Optional, Tuple

from gym.envs.classic_control.rendering import get_display, get_window
# as we've already imported gym's rendering we don't need to check
# if about OpenGL is available: gym's import would fail otherwise
import pyglet
from pyglet.gl import *
from pyglet.graphics import Batch, OrderedGroup
from pyglet.image import get_buffer_manager

# xxx(okachaiev): setup proper color pallet structure
black = (0,0,0)
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

player_colors = [blue, pink]

direction_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0),]
cell_direction_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# xxx(okachaiev): would be nice to have geometry over point
# with overloaded + and *
def translate_direction(direction, length):
    offset_x, offset_y = direction_offsets[direction]
    return (offset_x*length, offset_y*length)


# xxx(okachaiev): extend to all contstants
class ActionType:
    PRODUCE = 4


class Window:

    def __init__(self, width=640, height=640, title="MicroRTS", display=None):
        display = get_display(display)

        self._width = width
        self._height = height
        self._window = get_window(width=width, height=height, display=display)
        self._window.on_close = self._window_closed_by_user
        self._window.set_caption(title)
        self._open = True
        self._panels = []

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def add_panel(self, panel):
        panel.setup_viewport(self._width, self._height)
        self._panels.append(panel)

    def close(self):
        if self._open and sys.meta_path:
            for panel in self._panels:
                if hasattr(panel, "close"):
                    panel.close()
            self.window.close()
            self._open = False

    def _window_closed_by_user(self):
        self._open = False

    def render(self, return_rgb_array=False):
        glClearColor(1, 1, 1, 1)
        self._window.clear()
        self._window.switch_to()
        self._window.dispatch_events()

        for panel in self._panels:
            panel.on_render()

        rgb_array = None
        if return_rgb_array:
            buffer = get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            rgb_array = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            rgb_array = rgb_array.reshape(buffer.height, buffer.width, 4)
            rgb_array = rgb_array[::-1, :, 0:3]
        self._window.flip()
        return arr if return_rgb_array else self._open

    def __del__(self):
        # self.close()
        pass


class Tilemap:
    """Tilemap is a component responsible for rendering all
    given panels in a tiled view as closed to square configuration
    as possible.

    The component manages setup and rendering for child views, and
    is responsible for hanlding brining given tile into the focus
    by providing keypress and mouse click handlers.
    """

    def __init__(self, tiles):
        self._tiles = tiles
        self._initialized = False

    def setup_viewport(self, view_width, view_height):
        self._initialized = True
        pass

    def close(self):
        if self._initialized:
            for tile in self._tiles:
                tile.close()
        self._initialized = False
    
    def __del__(self):
        # self.close()
        pass


class GameStatePanel:

    def __init__(self, client, config=None):
        self._game_client = client
        self._game_config = config or {}

    # xxx(okachaiev): not sure if i need reference to the window
    def setup_viewport(self, view_width, view_height):
        self._width = view_width
        self._height = view_height
        self._init_grid()
        self._reset_canvas()

    def _init_grid(self):
        self._map_height, self._map_width = self._game_config["mapsize"]
        self._offset = 40
        self._xs, self._step = np.linspace(
            self._offset, self._width-self._offset, self._map_width+1, retstep=True
        )
        # xxx(okachaiev): this won't work for non-squared maps
        # i need to fit squares into min dimention and center to fit max dimension
        self._centers = self._xs[:-1]+self._step/2

    def _init_layers(self):
        self._batch = Batch()
        self._groups = {
            "background": OrderedGroup(0),
            "circle_foreground": OrderedGroup(1),
            "texts": OrderedGroup(2),
            "progress": OrderedGroup(3),
            "grid": OrderedGroup(4),
        }

    # xxx(okachaiev): it might be not the best approach to redefine
    # all geometries but it's a good enough starting point. later
    # i can reimplement it to track changes in already defined shapes
    # dropping entire batch seems brutal
    def _reset_canvas(self):
        # if hasattr(self, "_canvas"):
        #     for v in self._canvas:
        #         if hasattr(v, "delete"):
        #             v.delete()
        #     for label in self._labels:
        #         label.delete()
        self._canvas = []
        self._labels = []
        self._init_layers()
        self._add_grid_geom()
        self._add_info_bar_geom()

    def _add_to_canvas(self, *geoms):
        for geom in geoms:
            self._canvas.append(geom)

    def _add_label_to_canvas(self, *labels):
        for label in labels:
            self._labels.append(label)

    def _cell_to_coords(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        row, col = cell
        return self._centers[col-1], self._width-self._centers[row-1]

    def _add_grid_geom(self):
        for start in self._xs:
            hline = pyglet.shapes.Line(
                start, self._offset, start, self._width-self._offset,
                width=1, color=black,
                batch=self._batch, group=self._groups["grid"]
            )
            wline = pyglet.shapes.Line(
                self._offset, start, self._width-self._offset, start,
                width=1, color=black,
                batch=self._batch, group=self._groups["grid"]
            )
            self._add_to_canvas(hline, wline)

    # xxx(okachaiev): should have an option to add information for each
    # player when rendering (like different rewards they got so far)
    def _add_info_bar_geom(self):
        dot_radius = 6
        x, y = self._xs[0], self._xs[0] - self._offset/2
        for ind, player_name in enumerate(p["name"] for p in self._game_config["players"]):
            color = player_colors[ind]
            player_dot = pyglet.shapes.Circle(
                x+dot_radius, y, dot_radius,
                color=color,
                batch=self._batch, group=self._groups["grid"]
            )
            player_label = pyglet.text.Label(
                player_name,
                font_size=10,
                color=(0,0,0,255),
                bold=False,
                x=x+dot_radius*2+4, y=y,
                anchor_x="left", anchor_y="center",
                batch=self._batch, group=self._groups["grid"]
            )
            self._add_to_canvas(player_dot)
            self._add_label_to_canvas(player_label)
            x += dot_radius*2+4 + player_label.content_width + 8

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

    def _add_building_geom(self, cell, building_type, hp, max_hp, resources, owner):
        x, y = self._cell_to_coords(cell)
        color, = building_config[building_type]
        border_color = player_colors[owner]
        rect = pyglet.shapes.BorderedRectangle(
            x-self._step/2, y-self._step/2, self._step, self._step,
            border=3,
            color=color, border_color=border_color,
            batch=self._batch, group=self._groups["background"]
        )
        self._add_to_canvas(rect)
        text = self._add_resource_label_geom((x, y), resources)
        hp_progress = None
        if hp < max_hp:
            # xxx(okachaiev): progress bar for units could look like missing segment
            # rather than the bar on top of them
            hp_progress = self._add_progress_bar_geom((x,y), hp/max_hp, color, darkred)
        return rect, text, hp_progress

    def _add_unit_tick_geom(self, cell_coors, radius, direction, color):
        if direction is None: return None
        if direction > 3: return None
        x, y = cell_coors
        # xxx(okachaiev): if this is drawn first, there's no ned to compute
        # 2 pairs of offsets. just start from the mid of the cell
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
        border_color = player_colors[owner]
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
        color = player_colors[owner]
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

    def on_render(self):
        self._reset_canvas()

        gs = self._game_client.gs

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
                    unit.getPlayer(),
                )
            elif unit.getType().name in building_config:
                stockpile = 0
                if unit.getType().isStockpile:
                    stockpile = gs.getPlayer(unit.getPlayer()).getResources()
                # should be a building
                self._add_building_geom(
                    cell,
                    unit.getType().name,
                    unit.getHitPoints(),
                    unit.getMaxHitPoints(),
                    stockpile,
                    unit.getPlayer(),
                )
                if action is not None and action.action.getType() == ActionType.PRODUCE:
                    eta = action.time + action.action.ETA(action.unit) - gs.getTime()
                    progress = (1 - (eta / action.action.ETA(action.unit)))
                    label = str(action.action.getUnitType().name)
                    offset_row, offset_col = cell_direction_offsets[action.action.getDirection()]
                    action_cell = (unit.getY()+1+offset_row, unit.getX()+1+offset_col)
                    self._add_production_geom(action_cell, progress, label, unit.getPlayer())

        self._batch.draw()

    def close(self):
        self._reset_canvas()

    def __del__(self):
        # self.close()
        pass