from typing import Iterator, Optional
import math
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
import functools
import matplotlib.animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import gudhi
from gudhi import CubicalComplex, PeriodicCubicalComplex
import networkx as nx
from scipy.ndimage import label
from itertools import product
from pprint import pprint


class Position:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def __repr__(self):
        return str(self)

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Position") -> "Position":
        return Position(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __len__(self):
        """returns 0-norm of position"""
        return abs(self.x) + abs(self.y)

    def as_list(self) -> list[int]:
        return [self.x, self.y]


class Path:
    UP = Position(0, -1)
    RIGHT = Position(1, 0)
    DOWN = Position(0, 1)
    LEFT = Position(-1, 0)

    def __init__(self, path: list[Position]):
        self.path_points = path
        self.pos = path[0]
        self.dir_idx = 0
        self.dirs = [
            self.get_dir(p2 - p1)
            for p1, p2 in zip(path, path[1:] + [path[0]])
            if p1 != p2
        ]

    def __len__(self) -> int:
        return sum(
            len(p2 - p1)
            for p1, p2 in zip(
                self.path_points, self.path_points[1:] + [self.path_points[0]]
            )
        )

    def next(self) -> Position:
        self.pos = self.pos + self.dirs[self.dir_idx]

        next_dir_idx = (self.dir_idx + 1) % len(self.dirs)
        if self.pos == self.path_points[next_dir_idx]:
            self.dir_idx = next_dir_idx

        return self.pos

    def get_dir(self, pos: Position) -> Position:
        if pos.x > 0:
            return Path.RIGHT
        elif pos.x < 0:
            return Path.LEFT
        elif pos.y > 0:
            return Path.DOWN
        elif pos.y < 0:
            return Path.UP


class Sensor:
    def __init__(self, path: Path):
        self.path = path

    def __str__(self):
        return str(self.curr_pos)

    @property
    def curr_pos(self) -> Position:
        return self.path.pos

    @property
    def period(self) -> int:
        return len(self.path)

    def move(self):
        self.path.next()

    def slice(self) -> gudhi.cubical_complex.CubicalComplex:
        topleft = self.curr_pos + Position(-1, -1)
        # NOTE: topleft should just be position? (if we index vertices (0..=N, 0..=M) for NxM room)
        bottomright = self.curr_pos + Position(1, 1)

        return gudhi.cubical_complex.CubicalComplex(
            vertices=[topleft.as_list(), bottomright.as_list()]
        )


@dataclass
class SensorNetwork:
    name: str
    sensors: list[Sensor]
    room_width: int
    room_height: int

    def __post_init__(self):
        assert all(
            # TODO: Should we allow sensors moving along coordinate 0? (it is handled in cells_covered_by())
            0 <= p.x < self.room_width and 0 <= p.y < self.room_height
            for sensor in self.sensors
            for p in sensor.path.path_points
        ), "Path out of bounds"

    @functools.cached_property
    def period(self) -> int:
        return math.lcm(*[s.period for s in self.sensors])
    
    def __repr__(self) -> str:
        return f"Network \"{self.name}\" with {len(self.sensors)} sensors in {self.room_width}x{self.room_height} room (period {self.period})"

    def planar_slices(self) -> list[list[gudhi.cubical_complex.CubicalComplex]]:
        areas = []
        for _ in range(self.period):
            area = []
            for s in self.sensors:
                area.append(s.slice())
                s.move()

            areas.append(area)  # NOTE: could have duplicates (from overlapping sensors)

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
                if (
                    0 <= adjacent_cell_pos.x < self.room_width
                    and 0 <= adjacent_cell_pos.y < self.room_height
                ):
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
        return np.array(
            [
                [[i, j], [i + 1, j + 1]]
                for i in range(self.room_height)
                for j in range(self.room_width)
            ]
        )

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
                first_cell_free = not any(
                    contains_interval(s.vertices(), cell) for s in s1
                )
                second_cell_free = not any(
                    contains_interval(s.vertices(), cell) for s in s2
                )

                if first_cell_free and second_cell_free:
                    # TODO: Construct 3D cubical complex here
                    F_complex[(i, i + 1)].append(cell)

        return F_complex

    def evasion_complex(self) -> CubicalComplex:
        """Constructs the complex X * [0, p] where free cubes have filtration value 1 (and 0 otherwise).
        Indexing: (x, y, t)"""
        cube_filtration = np.zeros((self.room_width, self.room_height, self.period))

        covered_slices = list(self.covered_slices())

        # NOTE: easier to just use PeriodicCubicalComplex? (with periodic_dimensions=[F,F,T])
        for t in range(self.period):
            for cell in self.cell_positions:
                first_cell_free = cell not in covered_slices[t]
                second_cell_free = cell not in covered_slices[(t + 1) % self.period]

                if first_cell_free and second_cell_free:
                    cube_filtration[cell.x, cell.y, t] = 1

        # NOTE: using a vertices= constructor can lead to an undefined behavior in cofaces_of_persistence_pairs()
        return CubicalComplex(top_dimensional_cells=cube_filtration)

    def evasion_paths(self, compute_homology=True) -> list[list[Position]]:
        cpx = self.evasion_complex()
        cube_f = cpx.top_dimensional_cells() # filtration values of the top-dimensional cells (cubes)
        # print(f"{cpx.dimension()}-dim evasion complex with {cube_f.shape}-grid of cubes ({cpx.num_simplices()} simplices)")

        evasion_graph = collapse_to_graph(cpx)
        # print(evasion_graph)
        # draw_evasion_graph(evasion_graph)
    
        if compute_homology:
            collapsed_cpx = nx.Graph(evasion_graph)
            raise NotImplementedError("compute actual homology on undirected graph...")
            # NOTE: When computing homology, check the generators:
            #   generator could "time travel" or define a path which does not loop around along `p` - both
            #   cases do not represent a path thief can take
        else:
            # just find cyclic paths in the directed graph (should all be of length p due to the construction)
            starting_points = [node for node in evasion_graph.nodes if node[1] == 0] # position of thief in the first time interval
            paths = []

            path_buf = []
            def cyclic_paths_from(start, curr_node=None, depth_lim=self.period) -> Iterator[list[Position]]:
                if curr_node == start:
                    yield path_buf.copy()

                elif depth_lim > 0:
                    if curr_node is None: curr_node = start

                    path_buf.append(curr_node[0])
                    for succ in evasion_graph.successors(curr_node):
                        yield from cyclic_paths_from(start, succ, depth_lim - 1)

                    path_buf.pop()

            for start in starting_points:
                paths.extend(cyclic_paths_from(start))

            return paths
        
        cpx.compute_persistence()
        persistence_pairs, essential_features = cpx.cofaces_of_persistence_pairs()
        # could also try the vertices= constructor and then use cpx.vertices_of_persistence_pairs() ...

        for dim in range(max(len(persistence_pairs), len(essential_features))):
            print(f"homological dimension {dim}:")
            print("persistence pairs:")
            for pair in persistence_pairs[dim]:
                birth_idx, death_idx = (np.unravel_index(i, cube_f.shape) for i in pair)
                print(
                    f"\t{str(birth_idx):<11} [free={cube_f[birth_idx]}] -> {str(death_idx):<11} [free={cube_f[death_idx]}]"
                )

            if dim < len(essential_features):
                print("essential features:")
                for feature in essential_features[dim]:
                    birth_idx = np.unravel_index(feature, cube_f.shape)
                    print(f"\t{birth_idx} [free={cube_f[birth_idx]}]")

            print()


    def animate_movement(self, frametime=1, evasion_path: Optional[list[Position]] = None):
        """frames are separated by frametime seconds"""
        plt.ioff()
        fig, ax = plt.subplots()
        covered_at = list(self.covered_slices())
        # TODO: use evasion_path to highlight the path thief takes (arrows?)

        def animate(t):
            frame = np.zeros((self.room_height, self.room_width))
            for cell in covered_at[t]:
                frame[cell.y, cell.x] = 1
            plt.cla()
            plt.imshow(frame)
            if evasion_path is not None:
                pos_after = evasion_path[t]
                pos_before = evasion_path[t - 1]
                # print(f"t={t}: {pos_before} -> {pos_after}")
                plt.arrow(
                    pos_before.x,
                    pos_before.y,
                    pos_after.x - pos_before.x,
                    pos_after.y - pos_before.y,
                    color="red",
                    width=0.1,
                )

        ani = matplotlib.animation.FuncAnimation(
            fig, animate, frames=self.period, interval=1000 * frametime, repeat=True
        )
        plt.show()


def contains_interval(interval: np.ndarray, other: np.ndarray):
    return all(
        element[0] >= boundary[0] and element[1] <= boundary[1]
        for boundary, element in zip(interval.transpose(), other.transpose())
    )


def draw3d(cpx: CubicalComplex, free_full=True, alpha=0.5):
    cube_f = cpx.top_dimensional_cells()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    free_cpx = cube_f == 1
    ax.voxels(free_cpx if free_full else ~free_cpx, alpha=alpha)

    ax.set_xlim(0, cube_f.shape[0])
    ax.set_ylim(0, cube_f.shape[1])
    ax.set_zlim(0, cube_f.shape[2])
    set_axes_equal(ax)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("time")

    plt.show()


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def collapse_to_graph(cpx: CubicalComplex) -> nx.DiGraph:
    """Homotopy-preserving collapse to a graph (assuming no 3D caves).
    Nodes of the form (Position, t) where t is the start of a unit time interval.
    Edges oriented along the 3rd axis of the grid (time)."""

    voxels: np.ndarray = (cpx.top_dimensional_cells() == 1)
    p = voxels.shape[2]
    graph = nx.DiGraph()
    voxel_layer_comps: list[list[list[Position]]] = []

    for t in range(p):
        time_slice = voxels[:, :, t]
        labels, ncomps = label(time_slice) # find the connected components (by 4-connectivity)
        # print(f"t=[{t}, {t+1}]: {ncomps} components")
        comps = [[] for _ in range(ncomps)]

        for x, y in np.ndindex(labels.shape): # NOTE: first axis is x (room width)
            if (lab := labels[y, x]) != 0:
                comps[lab - 1].append(Position(x, y))

        voxel_layer_comps.append(comps)
        graph.add_nodes_from((comp[0], t) for comp in comps)
        # TODO: use the centroid as a component representative instead of arbitrary cell? (pretty sure these are convex components)

    for t in range(p):
        these_comps = voxel_layer_comps[t]
        next_comps = voxel_layer_comps[(t + 1) % p]

        for c1, c2 in product(these_comps, next_comps):
            if len(set(c1) & set(c2)) > 0: # OPT: convert all to set only once
                src = (c1[0], t)
                dst = (c2[0], (t + 1) % p)
                graph.add_edge(src, dst)

    return graph

def draw_evasion_graph(graph: nx.DiGraph):
    vert_loc = lambda pos: 10 * pos.x + pos.y # TODO: give access to room dimensions, use height + 1 instead of 10

    nx.draw(graph, with_labels=True, pos={node: (node[1], vert_loc(node[0])) for node in graph.nodes})
    plt.title(f"Evasion {graph}")
    plt.show()


NETWORKS = [
    SensorNetwork("instructions example", room_width=8, room_height=8, sensors=[
        Sensor(Path([Position(1, 1), Position(1, 6)])),
        Sensor(
            Path([Position(6, 1), Position(7, 1), Position(2, 1)]),
        ),
        Sensor(Path([Position(3, 5), Position(3, 3)])),
        Sensor(Path([Position(5, 4), Position(5, 5), Position(5, 3)])),
        Sensor(Path([Position(7, 5), Position(7, 7), Position(7, 3)])),
        Sensor(Path([Position(4, 7), Position(1, 7), Position(5, 7)]))
    ]),
    SensorNetwork("CCW circular sensor", room_width=4, room_height=4, sensors=[
        Sensor(Path([Position(1, 1), Position(1, 3), Position(3, 3), Position(3, 1)]))
    ]),
    # SensorNetwork("instructions fig. 4")
]

for network in NETWORKS[:1]:
    print(network)
    # draw3d(network.evasion_complex(), alpha=0.4)
    paths = network.evasion_paths(compute_homology=False)
    print(f"found {len(paths)} evasion paths")
    # pprint(paths)
    network.animate_movement(evasion_path=paths[0], frametime=1)
