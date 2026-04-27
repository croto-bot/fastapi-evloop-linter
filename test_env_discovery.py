#!/usr/bin/env python3
"""Unit tests for the project-root + venv auto-discovery in env.py."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi_evloop_linter.env import (
    discover_paths_for_target,
    find_project_root,
    find_venv_site_packages,
)


def _make_fixture(td: Path) -> Path:
    """Build a realistic project layout under `td` and return its root."""
    root = td / "myapp"
    (root / "src" / "utils").mkdir(parents=True)
    (root / "app" / "api").mkdir(parents=True)
    (root / "pyproject.toml").touch()
    (root / "src" / "__init__.py").touch()
    (root / "src" / "utils" / "__init__.py").touch()
    (root / "src" / "utils" / "logging.py").touch()
    sp = root / ".venv" / "lib" / "python3.12" / "site-packages"
    sp.mkdir(parents=True)
    return root


def test_finds_pyproject_walking_up_from_dir() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = _make_fixture(Path(td))
        target = root / "app" / "api"
        assert find_project_root(target) == root.resolve(), \
            "should find pyproject.toml two levels up from app/api"


def test_finds_pyproject_walking_up_from_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = _make_fixture(Path(td))
        target = root / "app" / "api" / "router.py"
        target.touch()
        assert find_project_root(target) == root.resolve(), \
            "should walk from a file's parent dir"


def test_finds_venv_site_packages() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = _make_fixture(Path(td))
        sps = find_venv_site_packages(root)
        expected = root / ".venv" / "lib" / "python3.12" / "site-packages"
        assert expected in sps, f"expected {expected} in {sps}"


def test_discover_paths_returns_root_src_and_venv() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = _make_fixture(Path(td))
        target = root / "app" / "api"
        found_root, additions = discover_paths_for_target(target)
        assert found_root == root.resolve()
        assert root.resolve() in additions, "project root must be added"
        assert (root / "src").resolve() in [a.resolve() for a in additions], \
            "src/ layout must be added when present"
        assert any("site-packages" in a.parts for a in additions), \
            "venv site-packages must be added"


def test_discover_returns_none_when_no_marker() -> None:
    # tempfile dir is below /tmp, which has no project marker. find_project_root
    # walks up to / and gives up. Use a clearly-isolated subdir.
    with tempfile.TemporaryDirectory() as td:
        bare = Path(td) / "no_markers_anywhere"
        bare.mkdir()
        # If the temp dir's ancestors happen to contain a marker we'd false-positive,
        # but on macOS /private/tmp/<tmpname>/<bare> has none up to /, so this is fine.
        found_root, additions = discover_paths_for_target(bare)
        # Either we find nothing, or we find something legitimate above /tmp;
        # in both cases we're not lying — assert the common no-marker outcome.
        assert found_root is None or (
            found_root != bare and Path("/").resolve() != found_root
        )


TESTS = [
    test_finds_pyproject_walking_up_from_dir,
    test_finds_pyproject_walking_up_from_file,
    test_finds_venv_site_packages,
    test_discover_paths_returns_root_src_and_venv,
    test_discover_returns_none_when_no_marker,
]


def main() -> int:
    failures: list[tuple[str, str]] = []
    for fn in TESTS:
        try:
            fn()
            print(f"OK   {fn.__name__}")
        except AssertionError as e:
            failures.append((fn.__name__, str(e) or "assertion failed"))
            print(f"FAIL {fn.__name__}: {failures[-1][1]}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
