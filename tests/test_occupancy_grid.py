"""The vectorised OccupancyGrid.update must equal the original per-cell loop, and be fast."""
import time
import unittest

import numpy as np

from src.navigation import OccupancyGrid


def reference_update(grid, origin, directions, ranges, max_range=5.):
    """The original scalar implementation (kept here as the specification)."""
    for direction, distance in zip(directions, ranges):
        hit = np.isfinite(distance)
        length = min(distance, max_range) if hit else max_range
        for t in np.arange(0., max(0., length-.02), grid.resolution/2):
            cell = grid.index(origin+direction*t)
            if grid.inside(cell) and grid.cells[cell] != 1:
                grid.cells[cell] = 0
        if hit:
            cell = grid.index(origin+direction*distance)
            if grid.inside(cell):
                grid.cells[cell] = 1


def random_scan(rng, rays, max_range=5.):
    angles = np.linspace(-np.pi, np.pi, rays, endpoint=False)
    directions = np.c_[np.cos(angles), np.sin(angles)]
    ranges = rng.uniform(.05, 1.4*max_range, rays)
    ranges[ranges > max_range] = np.nan  # misses, as the sensor reports
    ranges[rng.random(rays) < .02] = 0.   # degenerate zero-length returns
    return rng.uniform(-3, 1, 2), directions, ranges


class OccupancyGridUpdateTests(unittest.TestCase):
    def test_matches_the_original_loop_over_repeated_random_scans(self):
        rng = np.random.default_rng(3)
        for bounds, resolution in (((-6.5, 1.5, -3, 3), .04), ((0, 4, 0, 4), .1), ((-2.5, 1.5, -2, 2), .04)):
            fast, slow = OccupancyGrid(bounds, resolution), OccupancyGrid(bounds, resolution)
            for scan in range(6):  # state accumulates: occupied cells must persist
                origin, directions, ranges = random_scan(rng, int(rng.choice((90, 720, 1440))))
                fast.update(origin, directions, ranges, max_range=5.)
                reference_update(slow, origin, directions, ranges, max_range=5.)
                np.testing.assert_array_equal(fast.cells, slow.cells, (bounds, scan))

    def test_occupied_cells_are_never_cleared_and_unknown_stays_unknown(self):
        grid = OccupancyGrid((0, 4, 0, 4), .1)
        grid.cells[20, 20] = 1
        origin = np.array([.5, 2.05])
        grid.update(origin, np.array([[1., 0.]]), np.array([np.nan]), max_range=4.)  # ray through (2.05, 2.05)
        self.assertEqual(grid.cells[20, 20], 1)
        self.assertEqual(grid.cells[5, 20], 0)
        self.assertEqual(grid.cells[5, 35], -1)

    def test_a_1440_ray_scan_no_longer_stalls_the_simulation(self):
        rng = np.random.default_rng(5)
        grid = OccupancyGrid((-6.5, 1.5, -3, 3), .04)
        origin, directions, ranges = random_scan(rng, 1440, 6.5)
        started = time.perf_counter()
        grid.update(origin, directions, ranges, max_range=6.5)
        self.assertLess(time.perf_counter()-started, .5)  # the loop took about 1 s


if __name__ == '__main__':
    unittest.main()
