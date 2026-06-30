from pathlib import Path
import sys


def add_repo_root(file: str) -> None:
    script_dir = Path(file).resolve().parent
    repo_root = Path(file).resolve().parents[1]
    for path in (repo_root, script_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
