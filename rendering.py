import numpy as np

from gym.envs.classic_control.rendering import Viewer

viewer = Viewer(640, 640)
offset = 20
xs, step = np.linspace(offset, 640-offset, 17, retstep=True)
centers = xs[:-1]+step/2

green = (0,158,115)
yellow = (240,228,66)
blue = (0,114,178)
deepskyblue = (0,191,255)
pink = (204,121,167)
orange = (213,94,0)
silver = (192,192,192)
slategray = (112,128,144)

batch = pyglet.graphics.Batch()
background = pyglet.graphics.OrderedGroup(0)
cicle_foreground = pyglet.graphics.OrderedGroup(1)
texts = pyglet.graphics.OrderedGroup(2)

def cell_center_coords(cell):
    row, col = cell
    return centers[col-1], 640-centers[row-1]

def resource_label(resources, x, y, font_size=12):
    return pyglet.text.Label(
        str(resources),
        font_size=font_size,
        color=(0,0,0,255),
        bold=True,
        x=x, y=y,
        anchor_x="center", anchor_y="center",
        batch=batch, group=texts
    )

def resource_geom(cell, resources):
    x, y = cell_center_coords(cell)
    rect = pyglet.shapes.Rectangle(
        x-step/2, y-step/2, step, step,
        color=green,
        batch=batch, group=background
    )
    text = resource_label(resources, x, y)
    return (rect, text)

def building_geom(cell, building_type, hitpoints, production, resources, owner):
    x, y = cell_center_coords(cell)
    color = silver if building_type == "base" else slategray
    border_color = blue if owner == 1 else pink
    rect = pyglet.shapes.BorderedRectangle(
        x-step/2, y-step/2, step, step,
        border=2,
        color=color, border_color=border_color,
        batch=batch, group=background
    )
    text = resource_label(resources, x, y) if resources > 0 else None
    return rect, text

unit_config = {
    "worker": (silver, 0.6),
    "light": (yellow, 0.7),
    "heavy": (orange, 0.7),
    "range": (deepskyblue, 0.8),
}

def unit_geom(cell, unit_type, hitpoints, direction, resources, owner):
    x, y = cell_center_coords(cell)
    color, radius = unit_config[unit_type]
    border_color = blue if owner == 1 else pink
    back_circle = pyglet.shapes.Circle(
        x, y, radius*step/2+2,
        color=border_color,
        batch=batch, group=background
    )
    circle = pyglet.shapes.Circle(
        x, y, radius*step/2,
        color=color,
        batch=batch, group=cicle_foreground
    )
    text = resource_label(resources, x, y, font_size=9) if resources > 0 else None
    return back_circle, circle, text

def progress_bar(cell, progress, label_text, owner):
    x, y = cell_center_coords(cell)
    color = blue if owner == 1 else pink
    bar = pyglet.shapes.Rectangle(
        x-step/2, y+step/2-(0.2*step), step*progress, 0.2*step,
        color=color,
        batch=batch, group=background
    )
    text = pyglet.text.Label(
        label_text,
        font_size=8,
        color=color + (255,),
        bold=True,
        x=x, y=y,
        anchor_x="center", anchor_y="center",
        batch=batch, group=texts
    )
    return bar, text

class Proxy(Geom):

    def __init__(self, obj):
        self._obj = obj
    
    def render(self):
        self._obj.draw()

g1 = resource_geom((1,1), 25)
g2 = resource_geom((2,1), 20)
g3 = resource_geom((15,16), 20)
g4 = resource_geom((16,16), 25)

g5 = building_geom((3,3), "base", 3, None, 3, 1)
g6 = building_geom((14,14), "base", 3, None, 7, 2)
g7 = building_geom((3,7), "barracks", 3, None, 0, 1)

g8 = unit_geom((3,8), "worker", 0, None, 1, 1)
g9 = unit_geom((8,3), "light", 0, None, 0, 2)
g10 = unit_geom((12,12), "heavy", 0, None, 3, 2)
g11 = unit_geom((10,14), "range", 0, None, 0, 2)

g12 = progress_bar((9,9), 0.5, "worker", 1)
g13 = progress_bar((11,9), 0.8, "worker", 2)

viewer.add_geom(Proxy(batch))

for start in xs:
    viewer.add_geom(Line((start, 20), (start, 640-offset)))
    viewer.add_geom(Line((20, start), (640-offset, start)))

viewer.render()

while False:
    viewer.window.dispatch_events()
