"""Run model-written Python in a separate process.

Each call gets a fresh interpreter, a temporary working directory that is deleted
afterwards, an environment built from an allowlist (API keys and other variables of
the parent process are not passed on), a wall-clock timeout that kills the whole
process group, and a cap on the captured output.

This is process isolation with a timeout. It is not a security sandbox: the child runs
as the same user, can read and write any file that user can, and has network access.
Run untrusted prompts or files only inside a container or VM. POSIX only.
"""

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

CELL_FILENAME = "cell.py"
DATA_DIRNAME = "data"
OUTPUT_DIRNAME = "outputs"
READ_CHUNK_BYTES = 65_536
READER_JOIN_TIMEOUT_S = 5.0

# Runs inside the child. Executes cell.py and prints tracebacks without this wrapper's frame.
_RUNNER = """\
import sys
import traceback

with open("cell.py", encoding="utf-8") as source_file:
    source = source_file.read()
try:
    exec(compile(source, "cell.py", "exec"), {"__name__": "__main__"})
except SystemExit:
    raise
except BaseException as error:
    traceback.print_exception(type(error), error, error.__traceback__.tb_next)
    sys.exit(1)
"""


@dataclass(frozen=True)
class ExecutionLimits:
    timeout_s: float = 30.0
    max_output_bytes: int = 20_000


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int | None  # None when the process was killed at the timeout
    timed_out: bool
    output_truncated: bool
    duration_s: float
    artifacts: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


DEFAULT_LIMITS = ExecutionLimits()


def run_python(
    code: str,
    *,
    data_files: Iterable[Path] = (),
    artifact_dir: Path | None = None,
    limits: ExecutionLimits = DEFAULT_LIMITS,
    python: str = sys.executable,
) -> ExecutionResult:
    """Run `code` in a new process and return its captured output.

    Files in `data_files` are copied to `data/` in the working directory. Files the code
    writes to `outputs/` are moved to `artifact_dir` and listed in `artifacts`.
    """
    with tempfile.TemporaryDirectory(prefix="data-analyst-agent-run-") as tmp:
        workdir = Path(tmp)
        _prepare_workdir(workdir, code, data_files)
        started = time.monotonic()
        exit_code, timed_out, stdout, stderr = _run_child(workdir, python, limits)
        duration_s = time.monotonic() - started
        artifacts = ()
        if artifact_dir is not None:
            artifacts = _collect_artifacts(workdir / OUTPUT_DIRNAME, artifact_dir)
    return ExecutionResult(
        stdout=stdout.text,
        stderr=stderr.text,
        exit_code=exit_code,
        timed_out=timed_out,
        output_truncated=stdout.truncated or stderr.truncated,
        duration_s=duration_s,
        artifacts=artifacts,
    )


def minimal_env(workdir: Path) -> dict[str, str]:
    """Environment for the child process. Nothing is inherited from the parent."""
    tmp_dir = workdir / "tmp"
    return {
        "PATH": os.defpath,
        "HOME": str(workdir),
        "TMPDIR": str(tmp_dir),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_dir / "matplotlib"),
    }


def format_for_model(result: ExecutionResult, limits: ExecutionLimits) -> str:
    """Render a result as the tool message the model reads."""
    parts = []
    if result.stdout.strip():
        parts.append(result.stdout.rstrip())
    if result.stderr.strip():
        parts.append("[stderr]\n" + result.stderr.rstrip())
    if result.timed_out:
        parts.append(f"[timed out after {limits.timeout_s:g} s; the process was killed]")
    elif result.exit_code != 0:
        parts.append(f"[exit code {result.exit_code}]")
    if result.output_truncated:
        parts.append(
            f"[output truncated to the first {limits.max_output_bytes:,} bytes per stream]"
        )
    if result.artifacts:
        parts.append(f"[saved files] {', '.join(result.artifacts)}")
    return "\n".join(parts) or "[no output]"


def _prepare_workdir(workdir: Path, code: str, data_files: Iterable[Path]) -> None:
    (workdir / "tmp").mkdir()
    (workdir / OUTPUT_DIRNAME).mkdir()
    data_dir = workdir / DATA_DIRNAME
    data_dir.mkdir()
    for source in data_files:
        shutil.copy2(source, data_dir / Path(source).name)
    (workdir / CELL_FILENAME).write_text(code, encoding="utf-8")


def _run_child(
    workdir: Path, python: str, limits: ExecutionLimits
) -> tuple[int | None, bool, "_CappedReader", "_CappedReader"]:
    process = subprocess.Popen(
        [python, "-I", "-u", "-X", "utf8", "-c", _RUNNER],
        cwd=workdir,
        env=minimal_env(workdir),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    stdout = _CappedReader(process.stdout, limits.max_output_bytes)
    stderr = _CappedReader(process.stderr, limits.max_output_bytes)
    try:
        exit_code: int | None = process.wait(timeout=limits.timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        exit_code, timed_out = None, True
    # Kill the group in both cases so no background process outlives the call.
    _kill_process_group(process.pid)
    process.wait()
    stdout.join()
    stderr.join()
    return exit_code, timed_out, stdout, stderr


def _kill_process_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


class _CappedReader:
    """Drains a pipe on a thread, keeping at most `limit` bytes and counting the rest."""

    def __init__(self, stream: IO[bytes] | None, limit: int) -> None:
        self._stream = stream
        self._limit = limit
        self._kept = bytearray()
        self.total_bytes = 0
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        if self._stream is None:
            return
        with self._stream:
            while chunk := self._stream.read1(READ_CHUNK_BYTES):
                self.total_bytes += len(chunk)
                room = self._limit - len(self._kept)
                if room > 0:
                    self._kept += chunk[:room]

    def join(self) -> None:
        self._thread.join(READER_JOIN_TIMEOUT_S)

    @property
    def text(self) -> str:
        return bytes(self._kept).decode("utf-8", errors="replace")

    @property
    def truncated(self) -> bool:
        return self.total_bytes > self._limit


def _collect_artifacts(output_dir: Path, artifact_dir: Path) -> tuple[str, ...]:
    """Move regular files from outputs/ to artifact_dir. Symlinks and folders are skipped."""
    files = [p for p in sorted(output_dir.iterdir()) if p.is_file() and not p.is_symlink()]
    if not files:
        return ()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for path in files:
        target = _unique_path(artifact_dir, path.name)
        shutil.move(path, target)
        saved.append(target.name)
    return tuple(saved)


def _unique_path(directory: Path, name: str) -> Path:
    candidate = directory / name
    stem, suffix = Path(name).stem, Path(name).suffix
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate
