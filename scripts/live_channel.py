"""Locked shared snapshots between physics and its separate display process.

Only copied state arrays cross this boundary. Renderer edits never enter the
physical MjData or MjModel; speed/pause/stop are the explicit control channel.
"""
from contextlib import contextmanager
import fcntl
import mmap
import os
from pathlib import Path

import numpy as np


class LiveChannel:
    HEADER = 16
    FIELDS = {'simtime': 0, 'publishedwall': 1, 'speed': 2, 'paused': 3,
              'stop': 4, 'error': 5, 'finished': 6, 'ready': 7,
              'framecount': 8, 'p50': 9, 'p95': 10, 'max': 11}

    def __init__(self, path, model, create=False):
        self.path = Path(path)
        self.shapes = ((model.nq,), (model.nv,), (model.na,),
                       (model.nmocap, 3), (model.nmocap, 4))
        self.sizes = tuple(int(np.prod(shape)) for shape in self.shapes)
        self.length = self.HEADER + sum(self.sizes)
        self.fd = os.open(str(self.path), os.O_RDWR | (os.O_CREAT if create else 0), 0o600)
        try:
            if create:
                os.ftruncate(self.fd, self.length * 8)
            if os.fstat(self.fd).st_size != self.length * 8:
                raise ValueError('Live buffer dimensions do not match the display model')
            self.mapping = mmap.mmap(self.fd, self.length * 8)
            self.data = np.ndarray((self.length,), dtype=np.float64, buffer=self.mapping)
            if create:
                with self._locked():
                    self.data[:] = 0.
                    self.data[self.FIELDS['speed']] = 1.
        except BaseException:
            os.close(self.fd)
            raise

    @contextmanager
    def _locked(self, shared=False):
        fcntl.flock(self.fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def publish(self, state, now):
        values = (state.qpos, state.qvel, state.act, state.mocap_pos, state.mocap_quat)
        with self._locked():
            self.data[0:2] = (float(state.time), float(now))
            start = self.HEADER
            for array, size in zip(values, self.sizes):
                self.data[start:start+size] = np.asarray(array).reshape(-1)
                start += size

    def read_snapshot(self):
        with self._locked(shared=True):
            simulation_time, published = self.data[0:2].copy()
            arrays = []
            start = self.HEADER
            for shape, size in zip(self.shapes, self.sizes):
                arrays.append(self.data[start:start+size].copy().reshape(shape))
                start += size
        return (float(simulation_time), *arrays), float(published)

    def read_controls(self):
        with self._locked(shared=True):
            speed, paused, stop, error = self.data[2:6].copy()
        return float(speed), bool(paused), bool(stop), bool(error)

    def update(self, **kwargs):
        indices = [self.FIELDS[name] for name in kwargs]
        values = np.asarray(list(kwargs.values()), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Live controls must be finite')
        if 'speed' in kwargs and kwargs['speed'] <= 0:
            raise ValueError('Live speed must be positive')
        with self._locked():
            self.data[indices] = values

    def get(self, field):
        index = self.FIELDS[field]
        with self._locked(shared=True):
            return float(self.data[index])

    def close(self):
        if self.fd is not None:
            del self.data
            self.mapping.close()
            os.close(self.fd)
            self.fd = None


