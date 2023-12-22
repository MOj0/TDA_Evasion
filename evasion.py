from typing import TypeAlias
import math
import gudhi as gd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
import functools

# NOTE: alternative approach: Zig-zag persistent homology


class Position:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x if item == 0 else self.y

    def __setitem__(self, key, value):
        setattr(self, "x" if key == 0 else "y", value)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y)

    def as_list(self) -> list[int]:
        return [self.x, self.y]


Path: TypeAlias = tuple[Position, Position]


class Sensor:
    def __init__(self, start_at: Position, move_dir: Position, path: Path):
        self.curr_pos = start_at
        self.moving_component = 0 if move_dir.x != 0 else 1
        self.move_dir = move_dir[self.moving_component]
        self.path_min = min(
            path[0][self.moving_component],
            path[1][self.moving_component],
        )
        self.path_max = max(
            path[0][self.moving_component],
            path[1][self.moving_component],
        )

    def move(self):
        next_pos = self.curr_pos[self.moving_component] + self.move_dir
        if next_pos < self.path_min or next_pos > self.path_max:
            self.move_dir = -self.move_dir

        self.curr_pos[self.moving_component] += self.move_dir

    def area(self) -> list[Position]:
        return [
            self.curr_pos + Position(i, j) for i in range(-1, 2) for j in range(-1, 2)
        ]

    def slice(self) -> gd.cubical_complex.CubicalComplex:
        topleft = self.curr_pos + Position(-1, -1)
        bottomright = self.curr_pos + Position(1, 1)

        return gd.cubical_complex.CubicalComplex(
            vertices=[topleft.as_list(), bottomright.as_list()]
        )

    def __str__(self):
        return str(self.curr_pos)
    
    @property
    def period(self) -> int:
        return 2 * (self.path_max - self.path_min)
    

@dataclass
class SensorNetwork:
    # name: str
    sensors: list[Sensor]
    room_width: int
    room_height: int

    @functools.cached_property
    def period(self) -> int:
        return math.lcm(*[s.period for s in self.sensors])


    def planar_slices(self) -> list[list[gd.cubical_complex.CubicalComplex]]:
        areas = []
        for _ in range(self.period):
            area = []
            for s in self.sensors:
                area.append(s.slice()) 
                s.move()

            areas.append(area) # NOTE: could have overlapping covered areas

        return areas
    

    def all_cells(self) -> np.ndarray:
        # NOTE: a sm zamenou height pa width??
        return np.array([[[i, j], [i + 1, j + 1]] for i in range(self.room_height) for j in range(self.room_width)])
    

    def construct_F(self):
        r"""Constructs free subcomplex F = (X * [0, p] \ C)"""
        F_complex = defaultdict(list)
        slices = self.planar_slices()
        cells = self.all_cells()

        for i, (s1, s2) in enumerate(zip(slices, slices[1:] + [slices[0]])):
            for cell in cells:
                first_cell_free = not any(contains_interval(s.vertices(), cell) for s in s1)
                second_cell_free = not any(
                    contains_interval(s.vertices(), cell) for s in s2
                )

                if first_cell_free and second_cell_free:
                    # TODO: Construct 3D cubical complex here
                    F_complex[(i, i + 1)].append(cell)

        return F_complex
    

    def evasion_paths(self):
        # NOTE: When computing homology, check the generators:
        #   generator could "time travel" or define a path which does not loop around along `p` - both
        #   cases do not represent a path thief can take
        raise NotImplementedError



def contains_interval(interval: np.ndarray, other: np.ndarray):
    return all(
        element[0] >= boundary[0] and element[1] <= boundary[1]
        for boundary, element in zip(interval.transpose(), other.transpose())
    )


SAMPLE_SENSORS = [
    Sensor(Position(1, 1), Position(0, 1), Path((Position(1, 1), Position(1, 6)))),
    Sensor(Position(6, 1), Position(1, 0), Path((Position(2, 1), Position(7, 1)))),
    Sensor(Position(3, 5), Position(0, -1), Path((Position(3, 5), Position(3, 3)))),
    Sensor(Position(5, 4), Position(0, 1), Path((Position(5, 3), Position(5, 5)))),
    Sensor(Position(7, 5), Position(0, 1), Path((Position(7, 3), Position(7, 7)))),
    Sensor(Position(4, 7), Position(-1, 0), Path((Position(1, 7), Position(5, 7)))),
]

sample_network = SensorNetwork(SAMPLE_SENSORS, 8, 8)

print("p:", sample_network.period)
print("slices:")
for i, slice in enumerate(sample_network.planar_slices()):
    print(i, list(map(lambda s: s.vertices(), slice)))

print("\nF complex:")
print(sample_network.construct_F())
