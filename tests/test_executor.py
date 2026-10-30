import ast
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from data_analyst_agent.executor import (
    ExecutionLimits,
    ExecutionResult,
    _collect_artifacts,
    format_for_model,
    run_python,
)

FAST = ExecutionLimits(timeout_s=10.0, max_output_bytes=20_000)


def test_captures_stdout_and_exit_code():
    result = run_python("print('hello', 6 * 7)", limits=FAST)

    assert result.stdout == "hello 42\n"
    assert result.stderr == ""
    assert result.exit_code == 0
    assert result.succeeded
    assert not result.timed_out
    assert not result.output_truncated


def test_exception_is_reported_with_traceback_and_nonzero_exit():
    result = run_python("x = 1\nprint('before')\n1 / 0", limits=FAST)

    assert result.exit_code == 1
    assert not result.succeeded
    assert result.stdout == "before\n"
    assert 'File "cell.py", line 3' in result.stderr
    assert "ZeroDivisionError: division by zero" in result.stderr
    assert "exec(compile" not in result.stderr


def test_library_frames_do_not_show_installation_paths():
    result = run_python("import json\njson.loads('{')", limits=FAST)

    assert "json.decoder.JSONDecodeError" in result.stderr
    assert sys.prefix not in result.stderr
    assert sys.base_prefix not in result.stderr
    assert "<python>" in result.stderr or "<env>" in result.stderr


def test_syntax_error_is_reported():
    result = run_python("def broken(:\n    pass", limits=FAST)

    assert result.exit_code == 1
    assert "SyntaxError" in result.stderr


def test_sys_exit_in_generated_code_does_not_stop_the_caller():
    result = run_python("import sys\nsys.exit(3)", limits=FAST)

    assert result.exit_code == 3
    assert not result.succeeded


def test_timeout_kills_an_infinite_loop():
    limits = ExecutionLimits(timeout_s=1.0)
    started = time.monotonic()

    result = run_python("print('started')\nwhile True:\n    pass", limits=limits)

    assert result.timed_out
    assert result.exit_code is None
    assert result.stdout == "started\n"
    assert time.monotonic() - started < 6
    assert "[timed out after 1 s; the process was killed]" in format_for_model(result, limits)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_timeout_also_kills_child_processes():
    code = (
        "import subprocess, time\n"
        "child = subprocess.Popen(['sleep', '60'])\n"
        "print(child.pid)\n"
        "time.sleep(60)\n"
    )

    result = run_python(code, limits=ExecutionLimits(timeout_s=1.0))

    assert result.timed_out
    assert _wait_until_gone(int(result.stdout.strip()))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_background_process_is_killed_when_the_cell_finishes():
    code = (
        "import subprocess\n"
        "child = subprocess.Popen(['sleep', '60'], stdout=subprocess.DEVNULL)\n"
        "print(child.pid)\n"
    )

    result = run_python(code, limits=FAST)

    assert result.succeeded
    assert _wait_until_gone(int(result.stdout.strip()))


def test_environment_is_built_from_an_allowlist(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "parent-value-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "parent-value-2")
    monkeypatch.setenv("SOME_OTHER_SETTING", "parent-value-3")

    result = run_python("import os\nprint(sorted(os.environ))", limits=FAST)

    names = ast.literal_eval(result.stdout)
    assert "OPENAI_API_KEY" not in names
    assert "ANTHROPIC_API_KEY" not in names
    assert "SOME_OTHER_SETTING" not in names
    assert "parent-value" not in result.stdout
    assert set(names) <= {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "MPLBACKEND", "MPLCONFIGDIR"}


def test_each_call_gets_a_new_working_directory_that_is_removed():
    code = "import os\nprint(os.getcwd())"

    first = Path(run_python(code, limits=FAST).stdout.strip())
    second = Path(run_python(code, limits=FAST).stdout.strip())

    assert first != second
    assert first != Path.cwd()
    assert not first.exists()
    assert not second.exists()


def test_variables_do_not_persist_between_calls():
    run_python("value = 1", limits=FAST)

    result = run_python("print(value)", limits=FAST)

    assert "NameError: name 'value' is not defined" in result.stderr


def test_output_is_capped_per_stream():
    limits = ExecutionLimits(timeout_s=10.0, max_output_bytes=1_000)

    result = run_python(
        "import sys\nprint('x' * 50_000)\nsys.stderr.write('e' * 5_000)", limits=limits
    )

    assert len(result.stdout.encode()) == 1_000
    assert len(result.stderr.encode()) == 1_000
    assert result.output_truncated
    assert "[output truncated to the first 1,000 bytes per stream]" in format_for_model(
        result, limits
    )


def test_data_files_are_copies(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("a,b\n1,2\n")
    code = (
        "print(open('data/input.csv').read().strip())\n"
        "open('data/input.csv', 'w').write('overwritten')\n"
    )

    result = run_python(code, data_files=[source], limits=FAST)

    assert result.stdout == "a,b\n1,2\n"
    assert source.read_text() == "a,b\n1,2\n"


def test_files_saved_to_outputs_are_moved_to_the_artifact_dir(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    code = (
        "import os\n"
        "open('outputs/chart.html', 'w').write('<p>chart</p>')\n"
        "os.mkdir('outputs/folder')\n"
        "os.symlink('/etc/hostname', 'outputs/link.txt')\n"
    )

    first = run_python(code, artifact_dir=artifact_dir, limits=FAST)
    second = run_python(code, artifact_dir=artifact_dir, limits=FAST)

    assert first.artifacts == ("chart.html",)
    assert second.artifacts == ("chart-2.html",)
    assert sorted(p.name for p in artifact_dir.iterdir()) == ["chart-2.html", "chart.html"]
    assert (artifact_dir / "chart.html").read_text() == "<p>chart</p>"
    assert "[saved files] chart.html" in format_for_model(first, FAST)


def test_format_for_model_reports_stderr_and_exit_code():
    result = ExecutionResult(
        stdout="",
        stderr="Traceback\nKeyError: 'x'\n",
        exit_code=1,
        timed_out=False,
        output_truncated=False,
        duration_s=0.1,
        artifacts=(),
    )

    assert format_for_model(result, FAST) == "[stderr]\nTraceback\nKeyError: 'x'\n[exit code 1]"


def test_format_for_model_marks_empty_output():
    result = run_python("x = 1", limits=FAST)

    assert format_for_model(result, FAST) == "[no output]"


def test_functions_defined_in_the_cell_can_be_pickled():
    code = (
        "import pickle\n"
        "def square(x):\n"
        "    return x * x\n"
        "print(pickle.loads(pickle.dumps(square))(7))\n"
    )

    result = run_python(code, limits=FAST)

    assert result.stderr == ""
    assert result.stdout == "49\n"


def test_cell_runs_as_the_main_module():
    code = "import sys\nvalue = 5\nprint(__name__, sys.modules['__main__'].value)"

    result = run_python(code, limits=FAST)

    assert result.stdout == "__main__ 5\n"


def test_removed_outputs_folder_does_not_lose_the_result(tmp_path):
    code = "import shutil\nshutil.rmtree('outputs')\nprint(42)"

    result = run_python(code, artifact_dir=tmp_path / "artifacts", limits=FAST)

    assert result.succeeded
    assert result.stdout == "42\n"
    assert result.artifacts == ()


def test_outputs_replaced_by_a_symlink_is_not_followed(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    code = f"import os, shutil\nshutil.rmtree('outputs')\nos.symlink({str(outside)!r}, 'outputs')"

    result = run_python(code, artifact_dir=tmp_path / "artifacts", limits=FAST)

    assert result.artifacts == ()
    assert (outside / "keep.txt").read_text() == "keep"


def test_concurrent_collection_never_reuses_a_file_name(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    calls = 16
    output_dirs = []
    for index in range(calls):
        output_dir = tmp_path / f"outputs-{index}"
        output_dir.mkdir()
        (output_dir / "chart.html").write_text(str(index))
        output_dirs.append(output_dir)
    barrier = threading.Barrier(calls)
    saved: list[str] = []

    def collect(output_dir: Path) -> None:
        barrier.wait()
        saved.extend(_collect_artifacts(output_dir, artifact_dir))

    threads = [threading.Thread(target=collect, args=(d,)) for d in output_dirs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(set(saved)) == calls
    contents = sorted(int((artifact_dir / name).read_text()) for name in saved)
    assert contents == list(range(calls))


def _wait_until_gone(pid: int, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _is_running(pid):
            return True
        time.sleep(0.05)
    return False


def _is_running(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    state = stat.rsplit(")", 1)[1].split()[0]
    return state != "Z" and os.path.exists(f"/proc/{pid}")
