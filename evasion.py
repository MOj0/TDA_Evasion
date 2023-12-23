from typing import TypeAlias, Iterator
import math
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
import functools
import matplotlib.animation
import matplotlib.pyplot as plt

import gudhi
from gudhi import CubicalComplex, PeriodicCubicalComplex

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
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))


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

    def slice(self) -> gudhi.cubical_complex.CubicalComplex:
        topleft = self.curr_pos + Position(-1, -1)
        # NOTE: topleft should just be position? (if we index vertices (0..=N, 0..=M) for NxM room)
        bottomright = self.curr_pos + Position(1, 1)

        return gudhi.cubical_complex.CubicalComplex(
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

    def __post_init__(self):
        assert all(
            0 <= s.curr_pos.x < self.room_width and 0 <= s.curr_pos.y < self.room_height
            for s in self.sensors
        )
        # TODO: make sure the paths stay within the room

    @functools.cached_property
    def period(self) -> int:
        return math.lcm(*[s.period for s in self.sensors])

    def planar_slices(self) -> list[list[gudhi.cubical_complex.CubicalComplex]]:
        areas = []
        for _ in range(self.period):
            area = []
            for s in self.sensors:
                area.append(s.slice()) 
                s.move()

            areas.append(area) # NOTE: could have duplicates (from overlapping sensors)

        return areas
    
    def cells_covered_by(self, sensor: Sensor) -> Iterator[Position]:
        # the area covered by the sensors is the union of all the
        # unit squares that have one of the sensors at one of the vertices. 
        
        # NOTE: sensor Positions are grid vertices, not actual cells:
        # shifted coordinate system applies for cells, where vertex (x,y) is the top-left corner of the cell (x,y)
        # (assuming y axis points down)
        for dx in [-1, 0]:
            for dy in [-1, 0]:
                adjacent_cell_pos = sensor.curr_pos + Position(dx, dy)
                # any of these could be out of bounds (if sensor moving along the edge of the room)
                if 0 <= adjacent_cell_pos.x < self.room_width and 0 <= adjacent_cell_pos.y < self.room_height:
                    yield adjacent_cell_pos

    def covered_slices(self) -> Iterator[set[Position]]:
        """result[t] = positions of covered cells at time t"""
        for _ in range(self.period):
            covered_cells = set()
            for s in self.sensors:
                covered_cells.update(self.cells_covered_by(s))
                s.move()

            yield covered_cells

    def all_cells(self) -> np.ndarray:
        # NOTE: a sm zamenou height pa width?? (nima veze za sample)
        return np.array([[[i, j], [i + 1, j + 1]] for i in range(self.room_height) for j in range(self.room_width)])
    

    @property
    def cell_positions(self) -> Iterator[Position]:
        for x in range(self.room_width):
            for y in range(self.room_height):
                yield Position(x, y)
    

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
    

    def evasion_complex(self) -> CubicalComplex:
        """constructs the complex X * [0, p] where free cubes have filtration value 1 (and 0 otherwise)"""
        cube_filtration = np.zeros((self.room_width, self.room_height, self.period))

        covered_slices = list(self.covered_slices())

        # NOTE: easier to just use PeriodicCubicalComplex? (with periodic_dimensions=[F,F,T])
        for t in range(self.period):
            for cell in self.cell_positions:
                first_cell_free = cell not in covered_slices[t]
                second_cell_free = cell not in covered_slices[(t + 1) % self.period]

                if first_cell_free and second_cell_free:
                    # if t <= 1: print(f"free cell {cell} persists {t} -> {t+1}")
                    cube_filtration[cell.x, cell.y, t] = 1

        # NOTE: using a vertices= constructor can lead to an undefined behavior in cofaces_of_persistence_pairs() 
        return CubicalComplex(top_dimensional_cells=cube_filtration)


    def evasion_paths(self):
        # NOTE: When computing homology, check the generators:
        #   generator could "time travel" or define a path which does not loop around along `p` - both
        #   cases do not represent a path thief can take

        cpx = self.evasion_complex()
        cube_f = cpx.top_dimensional_cells() # filtration values of the top-dimensional cells (cubes)
        print(f"{cpx.dimension()}-dim evasion complex with {cube_f.shape}-grid of cubes ({cpx.num_simplices()} simplices)")

        cpx.compute_persistence()

        persistence_pairs, essential_features = cpx.cofaces_of_persistence_pairs()
        # 1st list: numpy arrays of shape (number_of_persistence_points, 2). The indices of the arrays in the list
        # correspond to the homological dimensions, and the integers of each row in each array correspond to:
        # (index of positive top-dimensional cell, index of negative top-dimensional cell).
        
        # The cells are represented by their indices in the input list of top-dimensional cells
        # (and not their indices in the internal datastructure that includes non-maximal cells).

        # 2nd list: the essential features, grouped by dimension. It contains numpy arrays
        # of shape (number_of_persistence_points,). The indices of the arrays in the list
        # correspond to the homological dimensions, and the integers of each row in each array correspond to:
        # (index of positive top-dimensional cell).

        for dim in range(max(len(persistence_pairs), len(essential_features))):
            print(f"homological dimension {dim}:")
            print("persistence pairs:") 
            for pair in persistence_pairs[dim]:
                birth_idx, death_idx = (np.unravel_index(i, cube_f.shape) for i in pair)
                print(f"\t{str(birth_idx):<11} [free={cube_f[birth_idx]}] -> {str(death_idx):<11} [free={cube_f[death_idx]}]")
            
            if dim < len(essential_features):
                print("essential features:")
                for feature in essential_features[dim]:
                    birth_idx = np.unravel_index(feature, cube_f.shape)
                    print(f"\t{birth_idx} [free={cube_f[birth_idx]}]")

            print()

        # could also try the vertices= constructor and then use cpx.vertices_of_persistence_pairs() ...
    

    def animate_coverage(self):
        plt.ioff()
        fig, ax = plt.subplots()
        covered_at = list(self.covered_slices())

        def animate(t):
            frame = np.zeros((self.room_height, self.room_width))
            for cell in covered_at[t]:
                frame[cell.y, cell.x] = 1
            plt.cla()
            plt.imshow(frame)

        ani = matplotlib.animation.FuncAnimation(fig, animate, frames=self.period, interval=1000 * 1, repeat=True)
        plt.show()



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

# print("p:", sample_network.period)
# print("slices:")
# for i, slice in enumerate(sample_network.planar_slices()):
#     print(i, list(map(lambda s: s.vertices(), slice)))

# print("\nF complex:")
# print(sample_network.construct_F())

# sample_network.animate_coverage()

sample_network.evasion_paths()