"""Auto-discover the user's project root and venv, then inject them into
sys.path before any module introspection runs.

When invoked via ``uvx --from "git+..." fastapi-evloop-linter app/`` the
linter executes inside an isolated venv where neither the user's
third-party dependencies (fastapi, requests, sqlalchemy, ...) nor their
project-local packages (src.utils.logging, app.api.schemas, ...) are
importable. Without this module the introspection layer would resolve
almost every user import to NOT_INSTALLED and the classifier would have
to fall back to brittle name-based heuristics.

Strategy: walk up from each linter target to find a project root, then
add (a) that root, (b) any conventional ``src/`` sibling, and (c) the
project's venv ``site-packages`` directories to sys.path. After that,
``importlib.util.find_spec(...)`` and ``importlib.import_module(...)``
work normally and the classifier can introspect classes, methods and
async-ness directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Files/directories whose presence marks the root of a Python project.
PROJECT_MARKERS: tuple[str, ...] = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    ".python-version",
    ".git",
)

# Conventional venv directory names, in priority order.
VENV_DIRS: tuple[str, ...] = (".venv", "venv", ".env")


def find_project_root(start: Path) -> Path | None:
    """Walk up from ``start`` to find a directory containing a project marker."""
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    while True:
        for marker in PROJECT_MARKERS:
            if (cur / marker).exists():
                return cur
        if cur.parent == cur:
            return None
        cur = cur.parent


def find_venv_site_packages(project_root: Path) -> list[Path]:
    """Locate site-packages directories under any conventional venv dir."""
    out: list[Path] = []
    for name in VENV_DIRS:
        venv = project_root / name
        if not venv.is_dir():
            continue
        # POSIX layout: <venv>/lib/python<major>.<minor>/site-packages
        lib = venv / "lib"
        if lib.is_dir():
            for child in sorted(lib.iterdir()):
                sp = child / "site-packages"
                if sp.is_dir():
                    out.append(sp)
        # Windows layout: <venv>/Lib/site-packages
        win = venv / "Lib" / "site-packages"
        if win.is_dir():
            out.append(win)
    return out


def discover_paths_for_target(target: Path) -> tuple[Path | None, list[Path]]:
    """Return ``(project_root, sys_path_additions)`` for one linter target."""
    root = find_project_root(target)
    if root is None:
        return None, []
    additions: list[Path] = [root]
    src = root / "src"
    if src.is_dir():
        additions.append(src)
    additions.extend(find_venv_site_packages(root))
    return root, additions


def setup_sys_path(targets: list[Path]) -> dict:
    """Insert each target's project root + venv site-packages into sys.path.

    Returns ``{"project_roots": [...], "site_packages": [...]}`` describing
    what was actually added (deduplicated), suitable for ``--show-env``.
    """
    seen: set[str] = set(sys.path)
    project_roots: list[str] = []
    site_packages: list[str] = []

    for target in targets:
        _root, additions = discover_paths_for_target(target)
        for path in additions:
            sp = str(path)
            if sp in seen:
                continue
            seen.add(sp)
            sys.path.insert(0, sp)
            if "site-packages" in path.parts:
                site_packages.append(sp)
            else:
                project_roots.append(sp)

    return {"project_roots": project_roots, "site_packages": site_packages}
