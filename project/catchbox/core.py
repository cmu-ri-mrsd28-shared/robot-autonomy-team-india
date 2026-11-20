"""Reusable SI-unit state, throws, prediction, and outcome accounting."""
from dataclasses import dataclass, asdict
from typing import Protocol
import numpy as np

@dataclass
class Config:
    timestep: float = 0.001
    control_dt: float = 0.01
    duration: float = 2.6
    radius: float = 0.02
    mass: float = 0.0027
    friction: tuple = (0.25, 0.005, 0.0001)
    solref: tuple = (0.006, 0.35)
    solimp: tuple = (0.95, 0.99, 0.001)
    box_width: float = 0.05
    box_depth: float = 0.05
    box_height: float = 0.12
    box_mass: float = 0.25
    printed_box: bool = False
    retention_seconds: float = 1.0

    def validate(self):
        for key in ('timestep', 'control_dt', 'duration', 'radius', 'mass',
                    'box_width', 'box_depth', 'box_height', 'box_mass', 'retention_seconds'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive and finite')
        if min(self.box_width, self.box_depth, self.box_height) <= 2*self.radius:
            raise ValueError('Box must fit the full ball')
        if self.duration < self.retention_seconds + 1:
            raise ValueError('Duration must allow flight plus retention')
        if self.solref[0] < 2*self.timestep or self.solref[1] <= 0:
            raise ValueError('solref needs time constant >= 2*timestep and positive damping')
        if any(x < 0 or not np.isfinite(x) for x in self.friction):
            raise ValueError('Friction must be finite and nonnegative')
        if not 0 < self.solimp[0] <= self.solimp[1] < 1 or self.solimp[2] <= 0:
            raise ValueError('Invalid solimp')
        if not np.isclose(self.control_dt/self.timestep, round(self.control_dt/self.timestep)):
            raise ValueError('control_dt must be an integer multiple of timestep')

@dataclass
class BallState:
    time: float
    position: np.ndarray
    velocity: np.ndarray
    valid: bool = True
    acceleration: np.ndarray | None = None

@dataclass
class RobotState:
    time: float
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    opening_position: np.ndarray
    opening_rotation: np.ndarray

class BallStateSource(Protocol):
    """Estimator seam: timestamped world-frame metres and metres/second."""
    def read(self) -> BallState: ...

@dataclass
class Throw:
    position: list
    velocity: list
    kind: str

def crossing(state: BallState, height: float, gravity=9.81):
    """Descending ballistic crossing; no contact, spin or aerodynamic model."""
    if not state.valid:
        return None
    z, vz = state.position[2]-height, state.velocity[2]
    disc = vz*vz+2*gravity*z
    if disc < 0:
        return None
    t = (vz+np.sqrt(disc))/gravity
    if t <= 0:
        return None
    p = state.position + state.velocity*t
    p[2] -= gravity*t*t/2
    return float(t), p

def sample_throws(seed, count, kind, opening):
    rng = np.random.default_rng(seed)
    throws = []
    for i in range(count):
        mode = ('drop' if i % 2 == 0 else 'lob') if kind == 'mixed' else kind
        target = np.array(opening, dtype=float)
        target[:2] += rng.uniform([-0.09, -0.07], [0.09, 0.07])
        if mode == 'drop':
            velocity = np.r_[rng.uniform(-0.03, 0.03, 2), 0.0]
            height = rng.uniform(0.35, 0.55)
        else:
            # Launch from in front of the robot, visibly upward and across the
            # workspace; avoid almost-vertical draws masquerading as lobs.
            angle = rng.uniform(-0.45, 0.45)
            speed = rng.uniform(0.8, 1.2)
            velocity = np.array([-speed*np.cos(angle), speed*np.sin(angle),
                                 rng.uniform(1.6, 2.2)])
            height = rng.uniform(-0.05, 0.10)
        flight = (velocity[2]+np.sqrt(velocity[2]**2+2*9.81*height))/9.81
        position = target - velocity*flight
        position[2] = target[2]+height
        throws.append(Throw(position.tolist(), velocity.tolist(), mode))
    return throws

class BoxMetrics:
    """Evaluated each physics step in the MOVING box frame; full-sphere clearance.

    Entry is a downward center crossing with the sphere clear of all four walls.
    Retention requires a continuous second within the cavity, allowing 1 mm
    lateral/top tolerance for numerical contact jitter. Rim contact is
    independent, so a rim-hit catch and a rim-hit bounce-out remain distinguishable.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.previous = None
        self.entered = False
        self.rim_hit = False
        self.contact = False
        self.escaped = False
        self.retained = False
        self.entry_time = None
        self.since = None
        self.max_retention = 0.0
        self.currently_outside = False

    def update(self, t, local, rim=False, contact=False):
        c = self.cfg
        self.rim_hit |= rim
        self.contact |= contact
        if self.previous is not None and self.previous[2] >= 0 > local[2]:
            fraction = self.previous[2] / (self.previous[2]-local[2])
            point = self.previous + fraction*(local-self.previous)
            if abs(point[0])+c.radius < c.box_width/2 and abs(point[1])+c.radius < c.box_depth/2:
                self.entered = True
                if self.entry_time is None:
                    self.entry_time = t
        tolerance = 0.001
        inside = (abs(local[0])+c.radius <= c.box_width/2+tolerance and
                  abs(local[1])+c.radius <= c.box_depth/2+tolerance and
                  -c.box_height+c.radius-0.002 <= local[2] <= -c.radius+tolerance)
        if self.entered and inside:
            if self.since is None:
                self.since = t
            self.max_retention = max(self.max_retention, t-self.since)
            self.retained |= self.max_retention >= c.retention_seconds-1e-9
        else:
            self.since = None
        if self.entered and local[2] > c.radius:
            self.escaped = True
        self.currently_outside = bool(local[2] > c.radius or
                                      abs(local[0]) > c.box_width/2+c.radius or
                                      abs(local[1]) > c.box_depth/2+c.radius)
        self.previous = np.array(local)

    def result(self):
        outcome = ('catch' if self.retained else 'entered_bounced_out' if self.escaped and self.currently_outside
                   else 'entered_not_retained' if self.entered else 'rim_hit_no_entry' if self.rim_hit
                   else 'other_contact_no_entry' if self.contact else 'miss')
        return dict(outcome=outcome, entered=self.entered, retained=self.retained,
                    rim_hit=self.rim_hit, escaped=self.escaped, contact=self.contact,
                    entry_time=self.entry_time, max_retention_s=self.max_retention)
