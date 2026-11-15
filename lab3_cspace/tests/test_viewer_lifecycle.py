"""Renderer teardown tests without requiring a display or importing MuJoCo."""
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import weakref
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from lab3_cspace.viewer_lifecycle import close_viewer


def test_close_waits_for_native_cleanup_not_just_exit_request():
    ready=threading.Event(); requested=threading.Event(); cleaned=threading.Event()
    handle=SimpleNamespace(close=requested.set)
    class NativeViewer:
        pass
    def renderer():
        owner=NativeViewer()
        handle._sim=weakref.ref(owner)
        ready.set()
        requested.wait()
        time.sleep(.03)  # Cleanup continues after public close() returns.
        cleaned.set()
        del owner
    worker=threading.Thread(target=renderer)
    worker.start()
    assert ready.wait(1.)
    close_viewer(handle)
    assert cleaned.is_set()
    assert handle._sim() is None
    worker.join(timeout=1.)
    assert not worker.is_alive()
    close_viewer(handle)  # Closing an already destroyed viewer is harmless.


def test_close_timeout_reports_failure_instead_of_claiming_completion():
    class NativeViewer:
        pass
    owner=NativeViewer()
    handle=SimpleNamespace(close=lambda:None,_sim=weakref.ref(owner))
    with pytest.raises(TimeoutError,match='renderer cleanup'):
        close_viewer(handle,timeout=0.)


def test_headless_close_needs_no_viewer():
    close_viewer(None)
