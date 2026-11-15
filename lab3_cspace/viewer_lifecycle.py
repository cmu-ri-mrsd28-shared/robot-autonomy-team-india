"""Finish passive-viewer teardown before Python/GLFW process cleanup."""
import time


def close_viewer(handle,timeout=10.):
    if handle is None:
        return
    # MuJoCo 3.x Handle.close() calls Simulate.exit(), which only requests
    # shutdown. is_running() becomes false immediately, before render_loop(),
    # its reload thread, and Simulate.destroy() have completed.
    # Handle exposes no public join operation. Its weak _sim reference expires
    # after _launch_internal has destroyed and released the native viewer.
    # Do not retain the referenced Simulate instance here: that would keep the
    # weak reference alive and prevent this completion check from succeeding.
    native_ref=getattr(handle,'_sim',None)
    handle.close()
    if native_ref is None or not callable(native_ref):
        raise RuntimeError('Unsupported MuJoCo viewer handle: cannot verify renderer shutdown')
    deadline=time.monotonic()+timeout
    while native_ref() is not None:
        if time.monotonic()>=deadline:
            raise TimeoutError('MuJoCo viewer did not finish renderer cleanup within 10 seconds')
        time.sleep(.01)
