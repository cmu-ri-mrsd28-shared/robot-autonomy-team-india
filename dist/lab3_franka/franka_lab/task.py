"""Compile the shared Lab 3 planner into hardware-independent block transfers."""

from dataclasses import asdict, dataclass

import numpy as np

from .patterns import PATTERNS
from .planner import plan_pattern


def vector(value, length, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite numbers")
    return result


def grid_xyz(config, cell):
    row, col = divmod(cell, 3)
    xy = ((1-row)*np.asarray(config['north_xy']) +
          (1-col)*np.asarray(config['west_xy']))*config['grid_spacing_m']
    return np.asarray(config['grid_center_grasp_tool_xyz']) + np.r_[xy, 0.]


def location_xyz(config, location):
    if location.startswith('grid:'):
        return grid_xyz(config, int(location.split(':')[1]))
    return np.asarray(config['staging_grasp_tool_xyz'][location.split(':')[1]], dtype=float)


def check_workspace(config, xyz):
    xyz = vector(xyz, 3, 'tool translation')
    lower = np.asarray(config['workspace_tool_min_xyz'])
    upper = np.asarray(config['workspace_tool_max_xyz'])
    if np.any(xyz < lower) or np.any(xyz > upper):
        raise ValueError(f"Tool waypoint {xyz.tolist()} is outside the configured workspace")
    return xyz


def validate_config(config, execute=False):
    if execute and config.get('calibrated') is not True:
        raise ValueError('Hardware execution requires measured calibration and calibrated=true; the example file is for dry runs')
    if not isinstance(config.get('robot_num'), int) or isinstance(config['robot_num'], bool) or config['robot_num'] < 1:
        raise ValueError('robot_num must be a positive integer')
    positive = ('grid_spacing_m', 'block_width_m', 'gripper_width_tolerance_m',
                'gripper_speed_m_s', 'grasp_force_N', 'max_translation_speed_m_s',
                'max_vertical_speed_m_s', 'max_rotation_speed_rad_s',
                'minimum_motion_duration_s', 'motion_timeout_margin_s',
                'translation_tolerance_m', 'rotation_tolerance_rad')
    for name in positive:
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not np.isfinite(config['pattern_hold_s']) or config['pattern_hold_s'] < 0:
        raise ValueError('pattern_hold_s must be finite and nonnegative')
    if type(config['use_impedance']) is not bool:
        raise ValueError('use_impedance must be a boolean')
    lower = vector(config['workspace_tool_min_xyz'], 3, 'workspace minimum')
    upper = vector(config['workspace_tool_max_xyz'], 3, 'workspace maximum')
    if np.any(lower >= upper):
        raise ValueError('Workspace minimum must be below maximum')
    north = vector(config['north_xy'], 2, 'north_xy')
    west = vector(config['west_xy'], 2, 'west_xy')
    if not np.isclose(np.linalg.norm(north), 1.) or not np.isclose(np.linalg.norm(west), 1.) or not np.isclose(north@west, 0.):
        raise ValueError('Grid axes must be orthogonal unit vectors')
    for axis, expected in (('NS', north), ('EW', west)):
        rot = np.asarray(config[f'tool_rotation_{axis}'], dtype=float)
        if rot.shape != (3,3) or not np.isfinite(rot).all() or not np.allclose(rot.T@rot, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rot), 1.):
            raise ValueError(f'{axis} requires a proper 3 by 3 rotation matrix, not a quaternion')
        if rot[2,2] > -0.99 or abs(rot[:2,1]@expected) < .99:
            raise ValueError(f'{axis} must be top-down with tool Y parallel to its finger-side axis')
    blocks = config['staging_grasp_tool_xyz']
    if set(blocks) != {f'{color}{i}' for color in 'RB' for i in range(4)}:
        raise ValueError('Exactly R0..R3 and B0..B3 staging positions are required')
    width = config['block_width_m']
    if config['grid_spacing_m'] <= width:
        raise ValueError('Grid pitch must exceed the block width')
    opening, grasp = config['gripper_open_width_m'], config['gripper_grasp_width_m']
    if not 0 <= grasp < width < opening <= .08:
        raise ValueError('Require 0 <= grasp width < block width < opening <= 0.08 m')
    if width - grasp <= 2*config['gripper_width_tolerance_m']:
        raise ValueError('Grasp command must be distinctly narrower than a held block, so an empty close fails width feedback')
    if not np.isfinite(config['transit_tool_z_m']):
        raise ValueError('Transit height must be finite')
    vector(config['grid_center_grasp_tool_xyz'], 3, 'grid center')
    all_points = [grid_xyz(config, i) for i in range(9)] + [vector(v,3,k) for k,v in blocks.items()]
    for point in all_points:
        check_workspace(config, point)
        check_workspace(config, [*point[:2],config['transit_tool_z_m']])
        if config['transit_tool_z_m'] - point[2] < .10:
            raise ValueError('Transit tool height must be at least 0.10 m above every grasp tool height')
    # A necessary clearance check; full hand/arm geometry still needs calibration.
    bank = list(blocks.values())
    for i, point in enumerate(bank):
        others = bank[i+1:] + [p.tolist() for p in all_points[:9]]
        if any(np.linalg.norm(np.asarray(point)[:2]-np.asarray(other)[:2]) < width+.01 for other in others):
            raise ValueError('Staging positions must be isolated from each other and the grid')
    check_workspace(config, config['preview_start_tool_xyz'])


@dataclass(frozen=True)
class Transfer:
    block: str
    source: str
    destination: str
    orientation: str


def compile_program(config, patterns, final_reset=False, test_transfer=False):
    validate_config(config)
    program = []
    locations = {name:f'home:{name}' for name in config['staging_grasp_tool_xyz']}
    previous = []
    for name in patterns:
        plan, visited = plan_pattern(PATTERNS[name])
        if plan is None:
            program.append({'kind':'skip','pattern':name,'visited_states':visited,
                            'reason':'No plan under the NS/EW top-grasp model'})
            continue
        if previous:
            for move in reversed(previous):
                inverse = Transfer(move.block, move.destination, move.source, move.orientation)
                program.append({'kind':'transfer',**asdict(inverse)})
                locations[move.block] = move.source
            program.append({'kind':'reset_complete'})
            previous = []
        for placement in plan:
            block = next(block for block in sorted(locations)
                         if block[0] == placement.color and locations[block].startswith('home:'))
            move = Transfer(block, locations[block], f'grid:{placement.cell}', placement.orientation)
            program.append({'kind':'transfer',**asdict(move)})
            locations[block] = move.destination
            previous.append(move)
            if test_transfer:
                inverse = Transfer(move.block, move.destination, move.source, move.orientation)
                return program + [{'kind':'transfer',**asdict(inverse)}, {'kind':'reset_complete'}]
        program.append({'kind':'pattern_commanded','pattern':name})
    if final_reset and previous:
        for move in reversed(previous):
            program.append({'kind':'transfer',**asdict(Transfer(move.block,move.destination,move.source,move.orientation))})
        program.append({'kind':'reset_complete'})
    return program
