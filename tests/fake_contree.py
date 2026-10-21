"""A local stand-in for contree-sdk's async client, for tests.

It mirrors the calls the executor makes (images.oci/use, image.run(...)
awaited into a new immutable image, image.read) and really executes the
commands in a temp dir, so branching semantics are exercised for real.
"""
import asyncio
import subprocess
import sys
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

WORK = "/workspace"


class FakeImage:
    def __init__(self, fs=None, registry=None):
        self.fs = dict(fs or {})          # immutable snapshot: path -> bytes
        self.uuid = uuid.uuid4()
        self.registry = registry
        self.exit_code, self.stdout, self.stderr = 0, "", ""
        self.result = SimpleNamespace(cost=0.0, elapsed_time=timedelta(0))
        self._pending = None
        if registry is not None:
            registry.append(self)

    def run(self, shell, files=None, disposable=True, timeout=None):
        child = FakeImage(self.fs, self.registry)
        child._pending = (shell, dict(files or {}))
        return child

    def __await__(self):
        return self._exec().__await__()

    async def _exec(self):
        shell, files = self._pending
        self.fs.update(files)
        with tempfile.TemporaryDirectory() as d:
            for path, data in self.fs.items():
                if path.startswith(WORK + "/"):
                    (Path(d) / path[len(WORK) + 1:]).write_bytes(data)
            if "pip install" in shell:
                proc = SimpleNamespace(returncode=0, stdout="", stderr="")
            else:
                cmd = shell.replace(WORK, d).replace("python ", f"{sys.executable} ")
                proc = await asyncio.to_thread(subprocess.run, cmd, shell=True, cwd=d,
                                               capture_output=True, text=True)
            for f in (x for x in Path(d).iterdir() if x.is_file()):
                self.fs[f"{WORK}/{f.name}"] = f.read_bytes()
        self.exit_code, self.stdout, self.stderr = proc.returncode, proc.stdout, proc.stderr
        self.result = SimpleNamespace(cost=0.0001, elapsed_time=timedelta(seconds=0.5))
        return self

    async def read(self, path):
        return self.fs[path]


class FakeImages:
    def __init__(self, registry):
        self.registry = registry

    async def oci(self, ref):
        return FakeImage(registry=self.registry)

    async def use(self, ref):
        return FakeImage(registry=self.registry)


class FakeContree:
    def __init__(self):
        self.created = []
        self.images = FakeImages(self.created)
