"""Remove only repository-generated experiment output directories."""

from pathlib import Path
import shutil


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    targets = [
        repo / "results" / "smoke",
        repo / "results" / "final_7001_7020",
        repo / "src" / "pqcho.egg-info",
        repo / "src" / "pqcho" / "__pycache__",
        repo / "scripts" / "__pycache__",
        repo / "tests" / "__pycache__",
    ]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
            print(f"Removed {target.relative_to(repo)}")


if __name__ == "__main__":
    main()
