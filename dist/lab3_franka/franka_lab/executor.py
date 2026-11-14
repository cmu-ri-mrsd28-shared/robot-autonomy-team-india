"""Calibrated Cartesian pick/place sequencing shared by dry-run and FrankaPy."""

import numpy as np

from .patterns import PATTERNS
from .planner import orientations
from .task import check_workspace, location_xyz


class Executor:
    def __init__(self, config, backend, progress=None):
        self.config = config
        self.backend = backend
        self.locations = {name:f'home:{name}' for name in config['staging_grasp_tool_xyz']}
        self.held_block = None
        self.phase = 'not started'
        self.transfers = []
        self.events = []
        self.state_uncertain = False
        self.progress = progress
        self.current_step = None
        self.total_transfers = 0
        self.motion_target = None

    def snapshot(self):
        return {'phase':self.phase,'current_step':self.current_step,
                'completed_transfers':len(self.transfers),'total_transfers':self.total_transfers,
                'nominal_locations':dict(self.locations),'held_block':self.held_block,
                'state_uncertain':self.state_uncertain,'motion_target':self.motion_target,
                'last_motion':getattr(self.backend,'last_motion',None)}

    def publish(self, **extra):
        if self.progress:
            self.progress(dict(self.snapshot(),**extra))

    def set_phase(self, label):
        self.phase = label
        self.motion_target = None
        print(f'    PHASE: {label}',flush=True)
        self.publish()

    def move(self, xyz, rotation, label, vertical=False):
        self.set_phase(label)
        xyz = check_workspace(self.config, xyz)
        current_xyz, current_rotation = self.backend.pose()
        check_workspace(self.config, current_xyz)
        distance = float(np.linalg.norm(xyz-current_xyz))
        angle = float(np.arccos(np.clip((np.trace(current_rotation.T@rotation)-1)/2,-1.,1.)))
        # Never request a substantial tool rotation while below the calibrated
        # transit height, or while a motion starts/ends below that height.
        # The small tolerance admits normal pose-feedback error, not a new yaw.
        clearance_floor = self.config['transit_tool_z_m']-self.config['translation_tolerance_m']
        if angle > self.config['rotation_tolerance_rad'] and min(current_xyz[2],xyz[2]) < clearance_floor:
            raise RuntimeError(f'Refusing tool rotation below transit height during {label}; raise vertically before rotating')
        speed = self.config['max_vertical_speed_m_s'] if vertical else self.config['max_translation_speed_m_s']
        # A minimum-jerk interpolant has peak speed 1.875 * distance/duration.
        duration = max(self.config['minimum_motion_duration_s'],1.875*distance/speed,
                       1.875*angle/self.config['max_rotation_speed_rad_s'])
        if distance > 1e-6 or angle > 1e-6:
            self.motion_target = {'translation':xyz.tolist(),'rotation':np.asarray(rotation).tolist(),
                                  'duration_s':float(duration)}
            self.publish()
            print(f'      target={np.round(xyz,4).tolist()}, duration={duration:.1f} s',flush=True)
            self.backend.move(xyz,rotation,duration,label)

    def _mask(self):
        return sum(1 << int(location.split(':')[1]) for location in self.locations.values()
                   if location.startswith('grid:'))

    def transfer(self, step):
        name, source, destination, axis = (step[k] for k in ('block','source','destination','orientation'))
        if self.held_block is not None or self.locations[name] != source:
            raise RuntimeError(f'Unexpected source state for {name}: {self.locations[name]}')
        if destination in self.locations.values():
            raise RuntimeError(f'Destination already occupied: {destination}')
        mask = self._mask()
        if source.startswith('grid:') and axis not in orientations(mask,int(source.split(':')[1])):
            raise RuntimeError('Blocked source grasp in the configured task model')
        if source.startswith('grid:'):
            mask &= ~(1 << int(source.split(':')[1]))
        if destination.startswith('grid:') and axis not in orientations(mask,int(destination.split(':')[1])):
            raise RuntimeError('Blocked destination grasp in the configured task model')
        source_xyz = location_xyz(self.config, source)
        destination_xyz = location_xyz(self.config, destination)
        rotation = np.asarray(self.config[f'tool_rotation_{axis}'],dtype=float)
        height = self.config['transit_tool_z_m']

        # First lift with the current orientation, then translate and rotate at height.
        current_xyz, current_rotation = self.backend.pose()
        clearance = max(height,current_xyz[2])
        self.move([*current_xyz[:2],clearance],current_rotation,'clearance lift',vertical=True)
        self.move([*source_xyz[:2],clearance],current_rotation,'above source')
        self.move([*source_xyz[:2],clearance],rotation,f'orient {axis} at clearance')
        self.move([*source_xyz[:2],height],rotation,'source transit height',vertical=True)
        self.set_phase('open above source')
        self.backend.gripper(False,self.phase)
        self.move(source_xyz,rotation,'descend to source',vertical=True)
        self.set_phase('grasp')
        self.backend.gripper(True,self.phase)
        self.held_block = name
        self.locations[name] = 'held'
        self.move([*source_xyz[:2],height],rotation,'lift block',vertical=True)
        self.set_phase('verify grasp after lift')
        self.backend.check_width(True,'after lift')
        self.move([*destination_xyz[:2],height],rotation,'transfer at clearance')
        self.set_phase('verify grasp before placement')
        self.backend.check_width(True,'before placement')
        self.move(destination_xyz,rotation,'descend to destination',vertical=True)
        self.set_phase('release')
        self.backend.gripper(False,self.phase)
        self.held_block = None
        self.locations[name] = destination
        self.move([*destination_xyz[:2],height],rotation,'vertical retreat',vertical=True)
        self.transfers.append(dict(step))
        print(f'    TRANSFER COMPLETE: {len(self.transfers)}/{self.total_transfers}',flush=True)
        self.publish()

    def check_nominal_pattern(self, pattern):
        actual = ['.']*9
        for block, location in self.locations.items():
            if location.startswith('grid:'):
                cell = int(location.split(':')[1])
                if actual[cell] != '.':
                    raise RuntimeError('Duplicate nominal grid occupancy')
                actual[cell] = block[0]
        if ''.join(actual) != ''.join(PATTERNS[pattern]):
            raise RuntimeError(f'Nominal task state does not match {pattern}')

    def run(self, program):
        self.total_transfers = sum(step['kind']=='transfer' for step in program)
        self.publish()
        try:
            for step in program:
                self.current_step = dict(step)
                if step['kind'] == 'transfer':
                    print(f"  TRANSFER {len(self.transfers)+1}/{self.total_transfers}: {step['block']}: {step['source']} -> {step['destination']} [{step['orientation']}]",flush=True)
                    self.transfer(step)
                elif step['kind'] == 'pattern_commanded':
                    self.check_nominal_pattern(step['pattern'])
                    self.backend.pause(self.config['pattern_hold_s'])
                    print(f"COMMANDS COMPLETED: {step['pattern']}",flush=True)
                    self.events.append(dict(step))
                elif step['kind'] == 'reset_complete':
                    if self.held_block is not None or any(v != f'home:{k}' for k,v in self.locations.items()):
                        raise RuntimeError('Nominal reset incomplete')
                    self.events.append(dict(step))
                elif step['kind'] == 'skip':
                    print(f"SKIP {step['pattern']}: {step['reason']}",flush=True)
                    self.events.append(dict(step))
                else:
                    raise ValueError(f'Unknown program step: {step}')
            self.set_phase('sequence completed')
        except BaseException as error:
            self.state_uncertain = True
            print(f'ABORTED during {self.phase}: {type(error).__name__}: {error}',flush=True)
            # Save the original failure before asking the SDK to stop; SDK stop
            # handling may itself wait for a faulted robot interface.
            try:
                self.publish(status='FAILED',error=f'{type(error).__name__}: {error}')
            except Exception:
                pass
            try:
                self.backend.stop()
            except BaseException as stop_error:
                print(f'Arm stop request also reported: {stop_error}',flush=True)
            raise
