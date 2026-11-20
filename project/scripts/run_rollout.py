"""Run one simulated throw with the saved printed-box policy."""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import CrossQ
from catchbox.learning.environment import CatchCore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42, help='Throw seed (default: 42)')
    parser.add_argument('--no-view', action='store_true', help='Run without opening the MuJoCo viewer')
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / 'artifacts/original-model/box/final_model.zip'
    model = CrossQ.load(str(checkpoint), device='cpu')
    env = CatchCore('box', seed=args.seed, printed_box=True)
    observation = env.reset(seed=args.seed)
    viewer = None
    if not args.no_view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.sim.model, env.sim.data)
        viewer.cam.lookat[:] = [.5, 0, .25]
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.opt.geomgroup[0] = 0
        viewer.opt.geomgroup[1] = 1
        viewer.opt.sitegroup[:] = 0

    try:
        while True:
            if viewer and not viewer.is_running():
                print('Viewer closed before the rollout finished.')
                return
            step_started = time.monotonic()
            action, _ = model.predict(observation, deterministic=True)
            observation, _, done, truncated, info = env.step(action)
            if viewer:
                viewer.sync()
                time.sleep(max(0.0, env.dt - (time.monotonic() - step_started)))
            if done or truncated:
                result = info['result']
                print(f"Outcome: {result['outcome']}")
                print(f"Entered: {result['entered']}; retained: {result['success']}")
                print(f"Simulated duration: {result['episode_duration_s']:.2f} s")
                if viewer:
                    print('Close the viewer to exit.')
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(.02)
                break
    finally:
        if viewer:
            viewer.close()


if __name__ == '__main__':
    main()
