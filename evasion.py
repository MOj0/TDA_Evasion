from typing import TypeAlias
import math

# NOTE: Zig-zag persistent homology...


class Position:
    def __init__(self, x, y):
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

    def __str__(self):
        return str(self.curr_pos)


SENSORS = [
    Sensor(Position(1, 1), Position(0, 1), Path((Position(1, 1), Position(1, 6)))),
    Sensor(Position(6, 1), Position(1, 0), Path((Position(2, 1), Position(7, 1)))),
    Sensor(Position(3, 5), Position(0, -1), Path((Position(3, 5), Position(3, 3)))),
    Sensor(Position(5, 4), Position(0, 1), Path((Position(5, 3), Position(5, 5)))),
    Sensor(Position(7, 5), Position(0, 1), Path((Position(7, 3), Position(7, 7)))),
    Sensor(Position(4, 7), Position(-1, 0), Path((Position(1, 7), Position(5, 7)))),
]


def compute_period(sensors: list[Sensor]) -> int:
    return math.lcm(*[2 * (s.path_max - s.path_min) for s in sensors])


def planar_slices(sensors: list[Sensor]) -> list[list[Position]]:
    areas = []
    p = compute_period(sensors)
    for _ in range(p):
        area = []
        for s in sensors:
            area.extend(s.area())
            s.move()

        areas.append(area)

    return areas


print("p:", compute_period(SENSORS))
for i, slice in enumerate(planar_slices(SENSORS)):
    print(i, list(map(str, slice)))
