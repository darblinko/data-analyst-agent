"""Per-session folders for the user's data files and the files the agent saves."""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    root: Path

    @classmethod
    def create(cls) -> "Workspace":
        root = Path(tempfile.mkdtemp(prefix="data-analyst-agent-session-"))
        workspace = cls(root)
        workspace.data_dir.mkdir()
        workspace.artifact_dir.mkdir()
        return workspace

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def artifact_dir(self) -> Path:
        return self.root / "artifacts"

    def data_files(self) -> tuple[Path, ...]:
        return tuple(sorted(p for p in self.data_dir.iterdir() if p.is_file()))

    def data_file_names(self) -> list[str]:
        return [p.name for p in self.data_files()]

    def add_data(self, name: str, content: bytes) -> Path:
        target = self.data_dir / safe_name(name)
        target.write_bytes(content)
        return target

    def copy_in(self, source: Path) -> Path:
        target = self.data_dir / safe_name(source.name)
        shutil.copyfile(source, target)
        return target

    def remove_data(self, name: str) -> None:
        (self.data_dir / safe_name(name)).unlink(missing_ok=True)

    def artifact_path(self, name: str) -> Path | None:
        """Path of a saved artifact, or None when no such file exists."""
        path = self.artifact_dir / safe_name(name)
        return path if path.is_file() else None


def safe_name(name: str) -> str:
    """Reduce a user or model supplied file name to a single path component."""
    base = Path(name.replace("\\", "/")).name
    if base in {"", ".", ".."}:
        raise ValueError(f"Invalid file name: {name!r}")
    return base
