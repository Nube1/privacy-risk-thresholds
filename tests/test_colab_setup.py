"""The sync helper is tested against a local bare repository, not against GitHub."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import colab_setup


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture()
def remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git("init", "--bare", "--initial-branch=main", ".", cwd=origin)
    seed = tmp_path / "seed"
    seed.mkdir()
    _git("init", "--initial-branch=main", ".", cwd=seed)
    (seed / "README.md").write_text("seed\n")
    _git("add", "-A", cwd=seed)
    _git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-m", "first",
         cwd=seed)
    _git("remote", "add", "origin", str(origin), cwd=seed)
    _git("push", "-u", "origin", "main", cwd=seed)
    monkeypatch.setenv("PRIVACY_RISK_REMOTE", str(origin))
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    return origin


def test_setup_clones_then_pulls(remote: Path, tmp_path: Path):
    target = tmp_path / "work"
    repo = colab_setup.setup(directory=str(target), install=False)
    assert (target / "README.md").exists()
    again = colab_setup.setup(directory=str(target), install=False)
    assert again.path == repo.path
    assert "main" in again.status()


def test_push_sends_work_to_the_remote(remote: Path, tmp_path: Path):
    repo = colab_setup.setup(directory=str(tmp_path / "work"), install=False)
    (repo.path / "notes.txt").write_text("a result\n")
    assert "nothing to commit" not in repo.push("add a result")
    other = tmp_path / "other"
    subprocess.run(["git", "clone", "--quiet", str(remote), str(other)], check=True)
    assert (other / "notes.txt").read_text() == "a result\n"


def test_push_with_nothing_staged_says_so(remote: Path, tmp_path: Path):
    repo = colab_setup.setup(directory=str(tmp_path / "work"), install=False)
    assert repo.push("no change") == "nothing to commit"


def test_pull_takes_remote_work_and_keeps_local_edits(remote: Path, tmp_path: Path):
    first = colab_setup.setup(directory=str(tmp_path / "first"), install=False)
    second = colab_setup.setup(directory=str(tmp_path / "second"), install=False)
    (first.path / "from_first.txt").write_text("one\n")
    first.push("work from the first copy")
    (second.path / "local.txt").write_text("uncommitted\n")
    second.pull()
    assert (second.path / "from_first.txt").exists()
    assert (second.path / "local.txt").read_text() == "uncommitted\n"


def test_push_without_a_token_explains_itself(remote: Path, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch):
    repo = colab_setup.setup(directory=str(tmp_path / "work"), install=False)
    (repo.path / "notes.txt").write_text("x\n")
    object.__setattr__(repo, "token", None)
    with pytest.raises(RuntimeError, match="No token"):
        repo.push("will not run")
