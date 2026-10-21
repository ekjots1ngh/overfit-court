"""A local stand-in for contree-sdk's async client, for tests.

It mirrors exactly the calls the executor makes: `images.oci()` / `images.use()`,
`image.run(...)` awaited into a new immutable image, `image.read(path)`, and the
`exit_code`, `stdout`, `stderr` and `result` attributes. Commands really run, in
a temporary directory, so checkpoint-and-branch semantics are exercised for real.
"""

import asyncio
import subprocess
import sys
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

WORKDIR = "/workspace"


class FakeImage:
    def __init__(self, files=None, registry=None):
        self.files = dict(files or {})  # an immutable snapshot: path -> bytes
        self.uuid = uuid.uuid4()
        self.exit_code, self.stdout, self.stderr = 0, "", ""
        self.result = SimpleNamespace(cost=0.0, elapsed_time=timedelta(0))
        self._registry = registry
        self._pending = None
        if registry is not None:
            registry.append(self)

    def run(self, shell, files=None, **_sdk_options):
        """Like the SDK: returns a new image to await; this one is never modified."""
        child = FakeImage(self.files, self._registry)
        child._pending = (shell, dict(files or {}))
        return child

    def __await__(self):
        return self._execute().__await__()

    async def _execute(self):
        shell, uploads = self._pending
        self.files.update(uploads)
        with tempfile.TemporaryDirectory() as folder:
            for path, data in self.files.items():
                if path.startswith(WORKDIR + "/"):
                    (Path(folder) / path[len(WORKDIR) + 1 :]).write_bytes(data)
            if "pip install" in shell:  # dependencies are already installed locally
                process = SimpleNamespace(returncode=0, stdout="", stderr="")
            else:
                command = shell.replace(WORKDIR, folder).replace("python ", f"{sys.executable} ")
                process = await asyncio.to_thread(
                    subprocess.run, command, shell=True, cwd=folder, capture_output=True, text=True
                )
            for produced in Path(folder).iterdir():
                if produced.is_file():
                    self.files[f"{WORKDIR}/{produced.name}"] = produced.read_bytes()
        self.exit_code, self.stdout, self.stderr = (
            process.returncode,
            process.stdout,
            process.stderr,
        )
        self.result = SimpleNamespace(cost=0.0001, elapsed_time=timedelta(seconds=0.5))
        return self

    async def read(self, path):
        return self.files[path]


class FakeImages:
    def __init__(self, registry):
        self._registry = registry

    async def oci(self, _reference):
        return FakeImage(registry=self._registry)

    async def use(self, _reference):
        return FakeImage(registry=self._registry)


class FakeContree:
    def __init__(self):
        self.created = []  # every image ever created, in order
        self.images = FakeImages(self.created)
