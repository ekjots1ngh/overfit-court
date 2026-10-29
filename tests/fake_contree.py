"""A local stand-in for contree-sdk's async client, for tests.

It mirrors exactly the calls the executor makes: `images.oci()` / `images.use()`,
`image.run(...)` awaited into a new immutable image, `image.read(path)`, and the
`exit_code`, `stdout`, `stderr` and `result` attributes. Commands really run, in
a temporary directory, so checkpoint-and-branch semantics are exercised for real.

It can also lose branches, the way a network or API failure would:
`flaky=n` loses the first n branch runs (a retry then succeeds), and
`dead_config=key` loses every run of the branch holding that configuration.
`slow_config=key` makes the branch holding that configuration time out, the
way the real service reports a run past its time limit. `forgotten` holds image
references the service no longer has: running from one fails as not found.
"""

import asyncio
import json
import subprocess
import sys
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from contree_sdk.sdk.exceptions import ContreeImageNotFoundError, OperationTimedOutError

WORKDIR = "/workspace"


class FakeImage:
    def __init__(self, contree, files=None):
        self.contree = contree
        self.files = dict(files or {})  # an immutable snapshot: path -> bytes
        self.uuid = uuid.uuid4()
        self.exit_code, self.stdout, self.stderr = 0, "", ""
        self.result = SimpleNamespace(cost=0.0, elapsed_time=timedelta(0))
        self._pending = None
        self.missing = False  # the service has no such image
        contree.created.append(self)

    def run(self, shell, files=None, **_sdk_options):
        """Like the SDK: returns a new image to await; this one is never modified."""
        child = FakeImage(self.contree, self.files)
        child._pending = (shell, dict(files or {}))
        child.missing = self.missing
        return child

    def __await__(self):
        return self._execute().__await__()

    async def _execute(self):
        if self.missing:
            raise ContreeImageNotFoundError(image_ref=str(self.uuid))
        shell, uploads = self._pending
        self.contree.lose_if_unlucky(shell, uploads)
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
    def __init__(self, contree):
        self._contree = contree

    async def oci(self, _reference):
        return FakeImage(self._contree)

    async def use(self, reference):
        image = FakeImage(self._contree)
        image.missing = str(reference) in self._contree.forgotten
        return image


class FakeContree:
    def __init__(
        self,
        flaky: int = 0,
        dead_config: str | None = None,
        slow_config: str | None = None,
        forgotten=(),
    ):
        self.created = []  # every image ever created, in order
        self.forgotten = set(forgotten)
        self.images = FakeImages(self)
        self.flaky = flaky
        self.dead_config = dead_config
        self.slow_config = slow_config
        self.lost = 0  # branch runs lost so far

    def lose_if_unlucky(self, shell, uploads):
        """Raise like a dropped connection, or a time-out, for the branch runs chosen."""
        if "runner.py" not in shell:
            return
        job = json.loads(uploads[f"{WORKDIR}/job.json"])
        if self.slow_config in job["grid"]:
            raise OperationTimedOutError(operation_uuid=uuid.uuid4())
        unlucky = self.flaky > 0 or self.dead_config in job["grid"]
        if not unlucky:
            return
        if self.dead_config not in job["grid"]:
            self.flaky -= 1
        self.lost += 1
        raise ConnectionError("the sandbox connection dropped")
