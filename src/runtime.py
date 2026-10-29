"""Optional observer around the canonical physics step, shared by task stages."""
from contextlib import contextmanager
from contextvars import ContextVar
import mujoco

_observer = ContextVar("wheelchair_physics_observer",default=None)


class SimulationStopped(KeyboardInterrupt):
    """User stopped a live run; never record this as task completion."""


@contextmanager
def observe(observer):
    token = _observer.set(observer)
    try:
        yield observer
    finally:
        _observer.reset(token)


def step(model,state):
    observer = _observer.get()
    if observer is not None:
        observer.before_step(model,state)
    mujoco.mj_step(model,state)
    mujoco.mj_forward(model,state)
    if observer is not None:
        observer.after_step(model,state)
