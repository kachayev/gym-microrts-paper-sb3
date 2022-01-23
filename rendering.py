import math
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
from pyglet.shapes import BorderedRectangle, Circle, Line, Rectangle

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


class Canvas:

    """Canvas object provides an easy way to connect relative
    coordinates of a specific viewport to a global coordinate
    system of the window (technically, of a the parent view).

    Note that Canvas maintains the same axis as a window in general:
    given (x, y) position of the canvas is a position of bottom
    left corner (left-hand rule).
    """

    def __init__(self, x, y, width, height):
        self._offset_x = x
        self._offset_y = y
        self._width = width
        self._height = height

    @property
    def viewport(self):
        return self._width, self._height

    def relative(self, x, y):
        if x < 0:
            x = self._width + x
        if y < 0:
            y = self._height + y
        return self._offset_x + x, self._offset_y + y


class Subcanvas(Canvas):

    def __init__(self, parent, x, y, width, height):
        self._parent = parent
        super().__init__(x, y, width, height)

    def relative(self, x, y):
        selfx, selfy = super().relative(x, y)
        return self._parent.relative(selfx, selfy)


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
        # xxx(okachaiev): somewhat sloppy on my side,
        # by i don't have more complex use cases as of now
        # to put work into building a proper solution
        # setting offset to (0, 0) meaning that by default
        # panel goes into bottom left corner of the window.
        # this might be contr-intuitive in a way, but it's
        # consistent with how all other objects are positioned
        # (e.g. anchors for text labels, etc).
        # also, hierarchical offset requires each view to
        # deal with with the fact they might be shifted in
        # coordinates
        self._default_canvas = Canvas(0, 0, self._width, self._height)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def add_panel(self, panel):
        panel.setup_canvas(self._default_canvas)
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

    def on_tick(self):
        self._window.switch_to()
        self._window.dispatch_events()

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

    def setup_canvas(self, canvas):
        self._canvas = canvas
        self._width, self._height = canvas.viewport
        n_tiles = len(self._tiles)
        grid_size = math.ceil(n_tiles ** 0.5)
        cell_width, cell_height = self._width/float(grid_size), self._height/float(grid_size)
        x, y = canvas.relative(0, 0)
        mx, my = canvas.relative(self._width, self._height)
        cells = np.mgrid[x:mx:(grid_size+1)*1j, y:my:(grid_size+1)*1j][:, :-1, :-1].reshape(2, -1).T
        for ind, tile in enumerate(self._tiles):
            x, y = cells[ind]
            tile.setup_canvas(Canvas(x, y, cell_width, cell_height))
        self._initialized = True

    def close(self):
        if self._initialized:
            for tile in self._tiles:
                tile.close()
        self._initialized = False
    
    def on_render(self):
        for tile in self._tiles:
            tile.on_render()

    def __del__(self):
        # self.close()
        pass


class GameStatePanel:

    def __init__(self, client, config=None):
        self._game_client = client
        self._game_config = config or {}
        self._canvas = None

    def setup_canvas(self, canvas):
        # if given canvas is not squared, replacing it
        # with a subview that has equal dims and center within
        # a boundaries of originally requested one
        w, h = canvas.viewport
        if w != h:
            d = min(w, h)
            canvas = Subcanvas(canvas, (w-d)/2, (h-d)/2, d, d)
        self._canvas = canvas
        self._init_grid()
        self._reset_viewport()

    def _init_grid(self):
        self._map_height, self._map_width = self._game_config["mapsize"]
        self._offset = 40
        w, h = self._canvas.viewport
        self._grid_canvas = Subcanvas(self._canvas, self._offset, self._offset, w-self._offset*2, h-self._offset*2)
        gw, gh = self._grid_canvas.viewport
        # xxx(okachaiev): this won't work for non-square maps at all
        self._step = gw/float(self._map_width+1)

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
    def _reset_viewport(self):
        # if hasattr(self, "_canvas"):
        #     for v in self._canvas:
        #         if hasattr(v, "delete"):
        #             v.delete()
        #     for label in self._labels:
        #         label.delete()
        self._geoms = []
        self._labels = []
        self._init_layers()
        self._add_grid_geom()
        self._add_info_bar_geom()

    def _add_to_canvas(self, *geoms):
        for geom in geoms:
            self._geoms.append(geom)

    def _add_label_to_canvas(self, *labels):
        for label in labels:
            self._labels.append(label)

    def _cell_to_coords(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        """
        Game level representation of cells:
         * (x,y) stands for (column, row),
         * (0,0) is a top level corner of the map.
        """
        col, row = cell
        return self._grid_canvas.relative(self._step*col+self._step/2, -1*(self._step*row+self._step/2))

    def _add_grid_geom(self):
        w, h = self._grid_canvas.viewport
        xs = np.linspace(0, w, self._map_width+1)
        for x in xs:
            bl_x, bl_y = self._grid_canvas.relative(x, 0)
            tl_x, tl_y = self._grid_canvas.relative(x, h) # xxx(okachaiev): i really want "-0"
            wline = Line(
                bl_x, bl_y, tl_x, tl_y,
                width=1, color=black,
                batch=self._batch, group=self._groups["grid"]
            )
            self._add_to_canvas(wline)
        ys = np.linspace(0, h, self._map_height+1)
        for y in ys:
            l_x, l_y = self._grid_canvas.relative(0, y)
            r_x, r_y = self._grid_canvas.relative(w, y)
            hline = Line(
                l_x, l_y, r_x, r_y,
                width=1, color=black,
                batch=self._batch, group=self._groups["grid"]
            )
            self._add_to_canvas(hline)

    # xxx(okachaiev): should have an option to add information for each
    # player when rendering (like different rewards they got so far)
    def _add_info_bar_geom(self):
        dot_radius = 6
        x, y = self._canvas.relative(self._offset, self._offset/2)
        for ind, player_name in enumerate(p["name"] for p in self._game_config["players"]):
            color = player_colors[ind]
            player_dot = Circle(
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
        rect = Rectangle(
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
        rect = BorderedRectangle(
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
        line = Line(
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
        back_circle = Circle(
            x, y, radius*self._step/2+2,
            color=border_color,
            batch=self._batch, group=self._groups["background"]
        )
        circle = Circle(
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
            right_bar = Rectangle(
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
        self._batch.draw()
        return

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


if __name__ == "__main__":

    class EmptyPanel:
        """Just a sample panel to test out how tiling
        works for screens of different dimensions.
        """

        def setup_canvas(self, canvas):
            self._canvas = canvas
            self._batch = Batch()

            x1, y1 = self._canvas.relative(5, 5) # bottom left corner
            x2, y2 = self._canvas.relative(-5, -5) # top right corner
            self._lines = [
                Line(x1, y1, x2, y2, width=1, color=darkred, batch=self._batch),
                Line(x1, y2, x2, y1, width=1, color=darkred, batch=self._batch),
                Line(x1, y1, x1, y2, width=3, color=black, batch=self._batch),
                Line(x2, y1, x2, y2, width=3, color=black, batch=self._batch),
                Line(x1, y1, x2, y1, width=3, color=black, batch=self._batch),
                Line(x1, y2, x2, y2, width=3, color=black, batch=self._batch),
            ]

        def on_render(self):
            self._batch.draw()

    window = Window(1024, 768)

    tiles = [EmptyPanel() for _ in range(20)]
    window.add_panel(Tilemap(tiles))

    window.render()

    while True:
        window.on_tick()