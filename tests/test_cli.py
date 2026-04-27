from __future__ import annotations

from pathlib import Path

from fastapi_evloop_linter.cli import expand_lint_paths


def test_app_target_expands_to_sibling_src(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n")
    app_dir = tmp_path / "app"
    src_dir = tmp_path / "src"
    app_dir.mkdir()
    src_dir.mkdir()

    assert expand_lint_paths([app_dir]) == [app_dir, src_dir]


def test_src_target_does_not_expand_to_app(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n")
    app_dir = tmp_path / "app"
    src_dir = tmp_path / "src"
    app_dir.mkdir()
    src_dir.mkdir()

    assert expand_lint_paths([src_dir]) == [src_dir]
