from collections import Counter
from dataclasses import dataclass
import math
import numpy as np
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from gym.envs.classic_control.rendering import get_display, get_window
# as we've already imported gym's rendering we don't need to check
# if about OpenGL is available: gym's import would fail otherwise
import pyglet
from pyglet.gl import *
from pyglet.graphics import Batch, OrderedGroup
from pyglet.image import get_buffer_manager
from pyglet.shapes import BorderedRectangle, Circle, Line, Rectangle

# xxx(okachaiev): setup proper color pallet structure
class Colors:
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
    "Worker": (0.5,),
    "Light": (0.6,),
    "Heavy": (0.6,),
    "Ranged": (0.8,),
}

direction_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0),]
cell_direction_offsets = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# xxx(okachaiev): would be nice to have geometry over point
# with overloaded + and *
def translate_direction(direction, length):
    offset_x, offset_y = direction_offsets[direction]
    return (offset_x*length, offset_y*length)

Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]

@dataclass
class Palette:
    gridline: Color
    resource: Color
    progress_hp: Color
    players: List[Color]
    units: Dict[str, Color]
    buildings: Dict[str, Color]

default_palette = Palette(
    gridline=Colors.black,
    resource=Colors.green,
    progress_hp=Colors.darkred,
    players=[Colors.blue, Colors.pink],
    units={
        "Worker": Colors.silver,
        "Light":  Colors.yellow,
        "Heavy":  Colors.orange,
        "Ranged": Colors.deepskyblue,
    },
    buildings={
        "Base":     Colors.silver,
        "Barracks": Colors.slategray,
    }
)

@dataclass
class GameStatePanelConfig:
    mapsize: Tuple[int, int]
    players: List[Dict[str, Any]]
    palette: Palette = default_palette

    @property
    def player_names(self) -> List[str]:
        return (p.get("name", "Unknown") for p in self.players)

default_game_state_panel_config = GameStatePanelConfig(
    mapsize=(16,16),
    players=[dict(name="Player #1"), dict(name="Player #2")],
)

# xxx(okachaiev): extend to all contstants
class ActionType:
    PRODUCE = 4
    ATTACK_LOCATION = 5

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
        self._geoms = []
        self._labels = []

    @property
    def viewport(self):
        return self._width, self._height

    def relative(self, x, y):
        if x < 0:
            x = self._width + x
        if y < 0:
            y = self._height + y
        return self._offset_x + x, self._offset_y + y

    def clear(self):
        self._geoms = []
        self._labels = []
    
    def add_geom(self, *geoms):
        self._geoms.extend(geoms)
    
    def add_label(self, *labels):
        self._labels.extend(labels)


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

    def add_panel(self, panel, *canvas_args):
        canvas = Canvas(*canvas_args) if canvas_args else self._default_canvas
        panel.setup_canvas(canvas)
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
        self._game_config = config or default_game_state_panel_config
        self._palette = self._game_config.palette
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

    @property
    def _gs(self):
        return self._game_client.gs

    def _init_grid(self):
        self._map_height, self._map_width = self._game_config.mapsize
        self._offset = 40
        w, h = self._canvas.viewport
        self._grid_canvas = Subcanvas(self._canvas, self._offset, self._offset, w-self._offset*2, h-self._offset*2)
        gw, gh = self._grid_canvas.viewport
        # xxx(okachaiev): this won't work for non-square maps at all
        self._step = gw/float(self._map_width)

    def _init_layers(self):
        self._batch = Batch()
        group_names = [
            "background",
            "ticks",
            "circle_foreground",
            "unit_sprites",
            "texts",
            "progress",
            "grid"
        ]
        self._groups = {n:OrderedGroup(ind) for ind, n in enumerate(group_names)}

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
        self._canvas.clear()
        self._init_layers()
        self._add_grid_geom()
        self._add_info_bar_geom()

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
                width=1, color=self._palette.gridline,
                batch=self._batch, group=self._groups["grid"]
            )
            self._canvas.add_geom(wline)
        ys = np.linspace(0, h, self._map_height+1)
        for y in ys:
            l_x, l_y = self._grid_canvas.relative(0, y)
            r_x, r_y = self._grid_canvas.relative(w, y)
            hline = Line(
                l_x, l_y, r_x, r_y,
                width=1, color=self._palette.gridline,
                batch=self._batch, group=self._groups["grid"]
            )
            self._canvas.add_geom(hline)

    def _units_per_player(self):
        if not self._gs: return {}
        return Counter(unit.getPlayer() for unit in self._gs.getUnits())

    # xxx(okachaiev): should have an option to add information for each
    # player when rendering (like different rewards they got so far)
    def _add_info_bar_geom(self):
        dot_radius = 6
        x, y = self._canvas.relative(self._offset, self._offset/2)
        unit_stats = self._units_per_player()
        for ind, player_name in enumerate(self._game_config.player_names):
            color = self._palette.players[ind]
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
            self._canvas.add_geom(player_dot)
            self._canvas.add_label(player_label)
            x += dot_radius*2+4 + player_label.content_width + 4
            unit_label = pyglet.text.Label(
                f"( {unit_stats.get(ind, 0)} )",
                font_size=10,
                color=(0,0,0,255),
                bold=False,
                x=x, y=y,
                anchor_x="left", anchor_y="center",
                batch=self._batch, group=self._groups["grid"]
            )
            x += unit_label.content_width + 8
            self._canvas.add_label(unit_label)

        if self._gs:
            time_label = pyglet.text.Label(
                f"T: {self._gs.getTime()}",
                font_size=10,
                color=(0,0,0,255),
                bold=False,
                x=x+4, y=y,
                anchor_x="left", anchor_y="center",
                batch=self._batch, group=self._groups["grid"]
            )
            self._canvas.add_label(time_label)

    def _add_resource_label_geom(self, cell_coords, resources, font_size=10):
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
        self._canvas.add_label(geom)
        return geom

    def _add_resource_geom(self, cell, resources):
        if resources == 0: return None
        x, y = self._cell_to_coords(cell)
        rect = Rectangle(
            x-self._step/2, y-self._step/2, self._step, self._step,
            color=self._palette.resource,
            batch=self._batch, group=self._groups["background"]
        )
        self._canvas.add_geom(rect)
        text = self._add_resource_label_geom((x, y), resources)
        return (rect, text)

    def _add_building_geom(self, cell, building_type, hp, max_hp, resources, action, owner):
        x, y = self._cell_to_coords(cell)
        color = self._palette.buildings[building_type]
        border_color = self._palette.players[owner]
        rect = BorderedRectangle(
            x-self._step/2, y-self._step/2, self._step, self._step,
            border=3,
            color=color, border_color=border_color,
            batch=self._batch, group=self._groups["background"]
        )
        self._canvas.add_geom(rect)
        text = self._add_resource_label_geom((x, y), resources)
        # progress bar for hitpoints
        hp_progress = None
        if hp < max_hp:
            # xxx(okachaiev): progress bar for units could look like missing segment
            # rather than the bar on top of them
            hp_progress = self._add_progress_bar_geom((x,y), hp/max_hp, color, self._palette.progress_hp)
        # new unit product
        prod = self._add_building_action_geom(cell, action, owner)
        return rect, text, hp_progress, prod

    def _add_building_action_geom(self, cell, action, owner):
        if action is None or action.action.getType() != ActionType.PRODUCE: return None
        eta = action.time + action.action.ETA(action.unit) - self._gs.getTime()
        progress = (1 - (eta / action.action.ETA(action.unit)))
        label = str(action.action.getUnitType().name)
        offset_x, offset_y = cell_direction_offsets[action.action.getDirection()]
        unit_x, unit_y = cell
        action_cell = (unit_x+offset_x, unit_y+offset_y)
        return self._add_production_geom(action_cell, progress, label, owner)

    def _add_unit_tick_geom(self, cell_coors, target, color):
        if target is None: return None
        x, y = cell_coors
        if isinstance(target, tuple):
            target_x, target_y = self._cell_to_coords(target)
            width = 1
            # color = color + (128,)
        elif target > 3:
            return None
        else:
            offset_x_hat, offset_y_hat = translate_direction(target, self._step)
            target_x, target_y = x+offset_x_hat, y+offset_y_hat
            width = 2
        line = Line(
            x, y, target_x, target_y,
            width=width, color=color,
            batch=self._batch, group=self._groups["ticks"]
        )
        self._canvas.add_geom(line)
        return line

    def _add_unit_geom(self, cell, unit_type, hp, max_hp, action, resources, owner):
        x, y = self._cell_to_coords(cell)
        radius, = unit_config[unit_type]
        color = self._palette.units[unit_type]
        border_color = self._palette.players[owner]
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
        self._canvas.add_geom(back_circle, circle)
        text = self._add_resource_label_geom((x, y), resources, font_size=9)
        tick = None
        if action is not None:
            if action.action.getType() == ActionType.ATTACK_LOCATION:
                target = (action.action.getLocationX(), action.action.getLocationY())
            else:
                target = action.action.getDirection()
            tick = self._add_unit_tick_geom((x,y), target, border_color)
        hp_progress = None
        if hp < max_hp:
            # xxx(okachaiev): progress bar for units could look like missing segment
            # rather than the bar on top of them
            hp_progress = self._add_progress_bar_geom((x,y), hp/max_hp, color, self._palette.progress_hp)
        return back_circle, circle, text, tick, hp_progress

    def _add_progress_bar_geom(self, cell_coords, progress, color, background_color):
        x, y = cell_coords
        left_bar = pyglet.shapes.Rectangle(
            x-self._step/2, y+self._step/2-(0.2*self._step), self._step*progress, 0.2*self._step,
            color=color,
            batch=self._batch, group=self._groups["progress"]
        )
        self._canvas.add_geom(left_bar)
        right_bar = None
        if background_color is not None:
            right_bar = Rectangle(
                x-self._step/2+self._step*progress, y+self._step/2-0.2*self._step,
                self._step*(1-progress), 0.2*self._step,
                color=background_color,
                batch=self._batch, group=self._groups["progress"]
            )
            self._canvas.add_geom(right_bar)
        return left_bar, right_bar

    def _add_production_geom(self, cell, progress, label_text, owner):
        x, y = self._cell_to_coords(cell)
        color = self._palette.players[owner]
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
        self._canvas.add_label(text)
        return bar, text

    def on_render(self):
        self._reset_viewport()

        for unit in self._gs.getUnits():
            action = self._gs.getActionAssignment(unit)
            cell = (unit.getX(), unit.getY())
            if unit.getType().isResource:
                self._add_resource_geom(cell, unit.getResources())
            elif unit.getType().canMove:
                self._add_unit_geom(
                    cell,
                    unit.getType().name,
                    unit.getHitPoints(),
                    unit.getMaxHitPoints(),
                    action,
                    unit.getResources(),
                    unit.getPlayer(),
                )
            elif unit.getType().name in self._palette.buildings:
                stockpile = 0
                if unit.getType().isStockpile:
                    stockpile = self._gs.getPlayer(unit.getPlayer()).getResources()
                # should be a building
                self._add_building_geom(
                    cell,
                    unit.getType().name,
                    unit.getHitPoints(),
                    unit.getMaxHitPoints(),
                    stockpile,
                    action,
                    unit.getPlayer(),
                )

        self._batch.draw()

    def close(self):
        self._reset_viewport()

    def __del__(self):
        # self.close()
        pass


class SpriteMap:

    def __init__(self, source_path, tilesize=(16,16), symbols=None):
        self._source = pyglet.image.load(source_path)
        self._w, self._h = self._source.width, self._source.height
        self._tw, self._th = tilesize
        self._regions = {}
        if symbols is not None:
            self.register_symbols(symbols)

    def _region_for_cell(self, row, col):
        return self._source.get_region(col*self._tw, self._h-(row+1)*self._th, self._tw, self._th)

    def register_symbols(self, symbols):
        for sym, cell in symbols.items():
            if isinstance(cell, tuple):
                self._regions[sym] = self._region_for_cell(*cell)
            else:
                self._regions[sym] = [self._region_for_cell(row, col) for row, col in cell]

    def get_region(self, sym, index=None):
        region = self._regions[sym]
        if not isinstance(region, list):
            return region
        elif index is not None:
            return region[index]
        else:
            return region[np.random.rand(len(region))]

    def create_sprite(self, sym, index=None, scale=None, **kwargs):
        geom = pyglet.sprite.Sprite(self.get_region(sym, index), **kwargs)
        if scale is not None:
            geom.scale = scale
        return geom

    @property
    def tilesize(self):
        return self._tw, self._th


class SpritePanel:

    def __init__(self, source_path, tilesize=(16,16), scale=1):
        self._tilesize = (16,16)
        self._scale = scale
        self._sprites = SpriteMap(
            source_path,
            tilesize=tilesize,
            # xxx(okachaiev): there are a lot of existing tile managers,
            # i don't think i don't need to invent a new way to split the
            # sprite or define a map
            symbols={
                "Water": [(24,0),(24,1),(24,2),(24,3)],
                "Walls": [(28,0),(28,1),(28,2),(29,0),(29,2),(30,0),(30,1),(30,2)],
                "Grass": [(0,0),(0,1),(0,2),(0,3)],
                "Trees": [(0,4),(0,5),(0,6)],
                "Worker": [(35,0),(36,0)],
                "Worker/Resources": [(35,2),(36,2)],
                "Light": [(16,1),(17,1)],
                "Heavy": [(16,3),(17,3)],
                "Ranged": [(16,0),(17,0)],
                "Base": [(2,0),(1,1),(2,1),(1,2),(2,2)],
                "Barracks": (4,6),
                "Water/Objects": [(27,6),(28,6),(34,6)],
            }
        )

    @property
    def sprite_map(self):
        return self._sprites

    def setup_canvas(self, canvas):
        self._canvas = canvas
        self._water = SpriteBackgroundPanel(self.sprite_map)
        self._water.setup_canvas(canvas)
        self._land = SpiteBattlefieldPanel(self.sprite_map)
        w, h = canvas.viewport
        land_canvas = Subcanvas(canvas, w-16*16/2, h-16*16/2, 16*16, 16*16)
        self._land.setup_canvas(land_canvas)

    def on_render(self):
        self._water.on_render()
        self._land.on_render()


class SpiteBattlefieldPanel:

    def __init__(self, sprites):
        self._sprites = sprites

    def setup_canvas(self, canvas):
        self._canvas = canvas
        self._batch = Batch()
        self._background_group = OrderedGroup(0)
        self._background_sprites = {}
        tw, th = self._sprites.tilesize
        for x in range(16):
            for y in range(16):
                posx, posy = self._canvas.relative(x*tw, y*th)
                self._background_sprites[(x,y)] = self._sprites.create_sprite(
                    "Grass", index=(x+y)%2,
                    x=posx, y=posy,
                    batch=self._batch, group=self._background_group
                )

    def on_render(self):
        self._batch.draw()


class SpriteBackgroundPanel:

    def __init__(self, sprites):
        self._sprites = sprites

    def setup_canvas(self, canvas):
        self._canvas = canvas
        self._batch = Batch()
        self._water_group = OrderedGroup(0)
        self._water_sprites = {}
        tw, th = self._sprites.tilesize
        for x in range(16):
            for y in range(16):
                posx, posy = self._canvas.relative(x*tw, y*th)
                self._water_sprites[(x,y)] = self._sprites.create_sprite(
                    "Water", index=(x+y)%2,
                    x=posx, y=posy,
                    batch=self._batch, group=self._water_group
                )

    def on_render(self):
        self._batch.draw()


# todo:
# * add more background (water, walls, ships, etc)
# * randomize grass with small objects, and possible roads
# * find a way to subscript resources and stockpiles (re-scale???)
# * use 2x2 grid instead of scale=2, visualize movement of units
# * mark attack with a special sprite, animate dead unit
# * change direction for sprites when units are moving around
class GameStateSpritePanel(GameStatePanel):

    def __init__(self, sprites, *args, **kwargs):
        self._sprites = sprites
        super().__init__(*args, **kwargs)

    def _add_grid_geom(self):
        pass

    def _add_resource_label_geom(self, *args, **kwargs):
        return None

    def _add_unit_geom(self, cell, unit_type, hp, max_hp, action, resources, owner):
        x, y = self._cell_to_coords(cell)
        geom = self._sprites.create_sprite(
            unit_type, index=owner, scale=2,
            # xxx(okachaiev): i bet this could be done with proper settings for anchor
            x=x-16, y=y-16,
            batch=self._batch, group=self._groups["unit_sprites"]
        )
        self._canvas.add_geom(geom)
        text = self._add_resource_label_geom((x, y), resources, font_size=9)
        tick = None
        if action is not None:
            if action.action.getType() == ActionType.ATTACK_LOCATION:
                target = (action.action.getLocationX(), action.action.getLocationY())
            else:
                target = action.action.getDirection()
            tick = self._add_unit_tick_geom((x,y), target, Colors.silver)
        hp_progress = None
        if hp < max_hp:
            # xxx(okachaiev): progress bar for units could look like missing segment
            # rather than the bar on top of them
            hp_progress = self._add_progress_bar_geom((x,y), hp/max_hp, color, Colors.darkred)
        return geom, text, tick, hp_progress

    def _add_resource_geom(self, cell, resources):
        if resources == 0: return None
        x, y = self._cell_to_coords(cell)
        geom = self._sprites.create_sprite(
            "Trees", index=int(x+y)%3, scale=2,
            # xxx(okachaiev): i bet this could be done with proper settings for anchor
            x=x-16, y=y-16,
            batch=self._batch, group=self._groups["background"]
        )
        self._canvas.add_geom(geom)
        text = self._add_resource_label_geom((x, y), resources)
        return (geom, text)

    def _add_building_geom(self, cell, building_type, hp, max_hp, resources, action, owner):
        x, y = self._cell_to_coords(cell)
        geom = self._sprites.create_sprite(
            building_type, index=owner, scale=2,
            # xxx(okachaiev): i bet this could be done with proper settings for anchor
            x=x-16, y=y-16,
            batch=self._batch, group=self._groups["background"]
        )
        self._canvas.add_geom(geom)
        text = self._add_resource_label_geom((x, y), resources)
        # progress bar for hitpoints
        hp_progress = None
        if hp < max_hp:
            # xxx(okachaiev): progress bar for units could look like missing segment
            # rather than the bar on top of them
            color = self._palette.buildings[building_type]
            hp_progress = self._add_progress_bar_geom((x,y), hp/max_hp, color, self._palette.progress_hp)
        # new unit product
        prod = self._add_building_action_geom(cell, action, owner)
        return geom, text, hp_progress, prod



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
                Line(x1, y1, x2, y2, width=1, color=Colors.darkred, batch=self._batch),
                Line(x1, y2, x2, y1, width=1, color=Colors.darkred, batch=self._batch),
                Line(x1, y1, x1, y2, width=3, color=Colors.black, batch=self._batch),
                Line(x2, y1, x2, y2, width=3, color=Colors.black, batch=self._batch),
                Line(x1, y1, x2, y1, width=3, color=Colors.black, batch=self._batch),
                Line(x1, y2, x2, y2, width=3, color=Colors.black, batch=self._batch),
            ]

        def on_render(self):
            self._batch.draw()

    window = Window(16*32, 16*32)

    def render_empty_panels():
        tiles = [EmptyPanel() for _ in range(20)]
        window.add_panel(Tilemap(tiles))


    # xxx(okachaiev): would be really nice to have "anchor=center" or similar
    window.add_panel(SpritePanel("resources/toen_medieval_strategy.png", scale=2), 40, 40, 16*32, 16*32)
    # window.add_panel(GameStateSpritePanel())
    window.render()

    while True:
        window.on_tick()