import shutil

import pytest

from data_analyst_agent.workspace import Workspace, safe_name


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("sales.csv", "sales.csv"),
        ("../../etc/passwd", "passwd"),
        ("/tmp/upload.xlsx", "upload.xlsx"),
        ("..\\..\\report.csv", "report.csv"),
    ],
)
def test_safe_name_keeps_only_the_last_path_component(given, expected):
    assert safe_name(given) == expected


@pytest.mark.parametrize("given", ["", ".", "..", "../", "a/.."])
def test_safe_name_rejects_names_without_a_file_component(given):
    with pytest.raises(ValueError):
        safe_name(given)


def test_data_files_are_added_listed_and_removed(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.data_dir.mkdir()

    workspace.add_data("../b.csv", b"x\n1\n")
    workspace.add_data("a.csv", b"x\n2\n")

    assert workspace.data_file_names() == ["a.csv", "b.csv"]
    assert (tmp_path / "data" / "b.csv").read_bytes() == b"x\n1\n"
    workspace.remove_data("b.csv")
    assert workspace.data_file_names() == ["a.csv"]


def test_artifact_path_only_returns_existing_files_inside_the_artifact_dir(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.artifact_dir.mkdir()
    (workspace.artifact_dir / "chart.html").write_text("<p>chart</p>")
    (tmp_path / "secret.txt").write_text("outside")

    assert workspace.artifact_path("chart.html") == workspace.artifact_dir / "chart.html"
    assert workspace.artifact_path("missing.png") is None
    assert workspace.artifact_path("../secret.txt") is None


def test_create_makes_the_session_folders():
    workspace = Workspace.create()
    try:
        assert workspace.data_dir.is_dir()
        assert workspace.artifact_dir.is_dir()
        assert workspace.data_file_names() == []
    finally:
        shutil.rmtree(workspace.root)
