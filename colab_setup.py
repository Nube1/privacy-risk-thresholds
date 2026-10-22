"""Clone, pull and push this repository from Colab or a shell without typing git commands.

``setup()`` clones the repository the first time and pulls every later time, installs the
package if asked, and moves into the repository root. ``Repo.push`` commits everything,
takes remote changes first and pushes. The GitHub token comes from the ``GITHUB_TOKEN``
environment variable (in Colab, a secret of that name). Nothing here prints the token or
writes it to a file: it is passed to git only for the one network call that needs it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REMOTE = "https://github.com/Nube1/privacy-risk-thresholds.git"
DEFAULT_DIRECTORY = "/content/privacy-risk-thresholds"
DRIVE_DIRECTORY = "/content/drive/MyDrive/privacy-risk-thresholds"
_IDENTITY = ("-c", "user.name=privacy-risk-colab", "-c", "user.email=colab@localhost")


def _token() -> str | None:
    """The token from the environment, or None when it is unset or empty."""
    value = os.environ.get("GITHUB_TOKEN", "").strip()
    return value or None


def _remote() -> str:
    return os.environ.get("PRIVACY_RISK_REMOTE", DEFAULT_REMOTE)


def _with_token(url: str, token: str | None) -> str:
    """An https GitHub URL with the token inserted; any other URL is returned unchanged."""
    if token and url.startswith("https://github.com/"):
        return url.replace("https://", f"https://x-access-token:{token}@", 1)
    return url


def _scrub(text: str, token: str | None) -> str:
    return text.replace(token, "***") if token else text


@dataclass(frozen=True)
class Repo:
    path: Path
    token: str | None = None
    use_git_credentials: bool = False

    def _git(self, *args: str, check: bool = True) -> str:
        result = subprocess.run(["git", *args], cwd=self.path, capture_output=True,
                                text=True)
        output = _scrub((result.stdout + result.stderr).strip(), self.token)
        if check and result.returncode != 0:
            raise RuntimeError(f"git {args[0]} failed: {output}")
        return output

    def _has_identity(self) -> bool:
        return bool(self._git("config", "user.email", check=False))

    def _remote_url(self) -> str:
        return _with_token(self._git("remote", "get-url", "origin"), self.token)

    def status(self) -> str:
        """Branch, how far it is from the remote, and the files that differ."""
        return self._git("status", "--short", "--branch")

    def pull(self) -> str:
        """Take remote changes, keeping uncommitted local edits in place."""
        identity = () if self._has_identity() else _IDENTITY
        branch = self._git("rev-parse", "--abbrev-ref", "HEAD")
        return self._git(*identity, "pull", "--rebase", "--autostash",
                         self._remote_url(), branch) or "up to date"

    def push(self, message: str) -> str:
        """Commit everything, rebase on the remote, push. Stops on a conflict."""
        if self.token is None and not self.use_git_credentials:
            raise RuntimeError("No token: set the GITHUB_TOKEN secret (see docs/COLAB.md), "
                               "or use scripts/sync.py to rely on your git credentials.")
        self._git("add", "-A")
        if not self._git("status", "--porcelain"):
            return "nothing to commit"
        identity = () if self._has_identity() else _IDENTITY
        self._git(*identity, "commit", "--quiet", "-m", message)
        self.pull()
        branch = self._git("rev-parse", "--abbrev-ref", "HEAD")
        self._git("push", "--quiet", self._remote_url(), f"HEAD:{branch}")
        return f"pushed to {branch}: {message}"


def setup(directory: str | None = None, install: bool = True, drive: bool = False) -> Repo:
    """Clone or pull, optionally install the package, and move into the repository."""
    if drive:  # pragma: no cover - needs Colab
        from google.colab import drive as colab_drive

        colab_drive.mount("/content/drive")
    target = Path(directory or (DRIVE_DIRECTORY if drive else DEFAULT_DIRECTORY))
    token = _token()
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(["git", "clone", "--quiet", _with_token(_remote(), token),
                                 str(target)], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"clone failed: {_scrub(result.stderr.strip(), token)}")
        # Keep the stored remote free of the token.
        subprocess.run(["git", "remote", "set-url", "origin", _remote()], cwd=target,
                       check=True)
        repo = Repo(path=target.resolve(), token=token)
    else:
        repo = Repo(path=target.resolve(), token=token)
        repo.pull()
    if install:  # pragma: no cover - slow, exercised in Colab
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-e", "."],
                       cwd=repo.path, check=True)
    os.chdir(repo.path)
    if str(repo.path) not in sys.path:
        sys.path.insert(0, str(repo.path))
    print(f"working in {repo.path}\n{repo.status()}")
    return repo
