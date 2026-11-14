"""Wall-clock pacing for the passive viewer, independent of physics settings."""

import math
import time


class ViewerTiming:
    def __init__(self, speed=1., fps=30., clock=time.monotonic, sleep=time.sleep):
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("Viewer speed must be finite and nonnegative (0 means unlimited)")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Viewer FPS must be finite and positive")
        self.speed = speed
        self.interval = 1. / fps
        self.clock = clock
        self.sleep = sleep
        self.wall_start = None
        self.sim_start = None
        self.next_update = None

    def update_due(self, sim_time):
        """Pace simulated time; request synchronization at most FPS times/second."""
        now = self.clock()
        if self.wall_start is None:
            self.wall_start = now
            self.sim_start = sim_time
            self.next_update = now
        if self.speed:
            target = self.wall_start + (sim_time - self.sim_start) / self.speed
            if target > now:
                self.sleep(target - now)
                now = self.clock()
        if now >= self.next_update:
            self.next_update = now + self.interval
            return True
        return False
