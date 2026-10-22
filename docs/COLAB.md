# Working from Colab without instructions

One token has to exist once. After that, every Colab session starts with one cell and
you never type a git command again.

## Once, and only once

1. On GitHub, open **Settings, Developer settings, Personal access tokens, Fine-grained
   tokens**, and create a token. Repository access: only `Nube1/privacy-risk-thresholds`.
   Permissions: **Contents, read and write**. Copy it.
2. In Colab, open the key icon in the left sidebar, add a secret named `GITHUB_TOKEN`,
   paste the token, and switch on **Notebook access**.

Colab keeps that secret across notebooks and sessions, so this does not come back.

## Every session, one cell

```python
import os, subprocess, sys
try:
    from google.colab import userdata
    os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
except Exception:
    pass
url = f"https://x-access-token:{os.environ.get('GITHUB_TOKEN','')}@github.com/Nube1/privacy-risk-thresholds.git"
subprocess.run(["git", "clone", "--quiet", url, "/content/privacy-risk-thresholds"],
               check=False)
sys.path.insert(0, "/content/privacy-risk-thresholds")
import colab_setup
repo = colab_setup.setup()
```

The cell clones the repository the first time and pulls every later time, installs the
package, and leaves you in the repository directory. It prints where it is and what has
changed.

## Sending work back

```python
repo.push("what changed")
```

That commits everything, takes any remote changes first, and pushes. If there is nothing
to commit it says so and does nothing. If someone else changed the same lines, it stops
and tells you, rather than making a mess.

```python
repo.pull()      # take remote changes, keeping whatever you have open
repo.status()    # what is different here
```

## Keeping the working copy in Drive

Colab machines are recycled, and with the setup above that costs nothing, because the
repository is the state and the next cell run rebuilds it. If you would rather keep the
files:

```python
repo = colab_setup.setup(drive=True)
```

This mounts Drive and works in `MyDrive/privacy-risk-thresholds`. The first run asks for
Drive permission.

## Running the pipeline from there

```python
!privacy-risk-run --config configs/base.json  --output artifacts        # seconds
!privacy-risk-run --config configs/paper.json --output artifacts-paper  # minutes
repo.push("refresh the evidence")
```

`artifacts/` is ignored by git, `artifacts-paper/` is tracked, so the second command is
the one whose results travel with the repository.

## On your own machine

Same behaviour without Colab:

```bash
export GITHUB_TOKEN=...            # or rely on your existing git credentials
python scripts/sync.py pull
python scripts/sync.py push -m "what changed"
```

## If the token is ever wrong

`repo.push` raises and names the secret it looked for. Nothing prints the token, and no
file in the repository holds it.
