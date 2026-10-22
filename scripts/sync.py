"""Pull and push this repository from a shell, with the same rules as Colab.

    python scripts/sync.py pull
    python scripts/sync.py push -m "what changed"
    python scripts/sync.py status
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import colab_setup  # noqa: E402 - the helper lives at the repository root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pull", "push", "status"])
    parser.add_argument("-m", "--message", default="update from the working copy")
    parser.add_argument("--path", default=str(Path(__file__).resolve().parent.parent))
    arguments = parser.parse_args()

    token = colab_setup._token()
    repo = colab_setup.Repo(path=Path(arguments.path), token=token,
                            use_git_credentials=token is None)
    if arguments.action == "pull":
        print(repo.pull())
    elif arguments.action == "status":
        print(repo.status())
    else:
        print(repo.push(arguments.message))
    return 0


if __name__ == "__main__":
    sys.exit(main())
