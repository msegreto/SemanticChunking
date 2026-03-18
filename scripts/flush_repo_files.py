#!/usr/bin/env python3

from __future__ import annotations

import argparse
import signal
from pathlib import Path


def collect_files(root_dir: Path, targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for rel_dir in targets:
        abs_dir = root_dir / rel_dir
        if abs_dir.is_dir():
            files.extend(
                sorted(
                    p
                    for p in abs_dir.rglob("*")
                    if p.is_file() and p.name.lower() != "readme.md"
                )
            )
    return files


def main() -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    parser = argparse.ArgumentParser(
        description=(
            "Delete all files inside configs/experiments, data, and results, "
            "while preserving directory structure."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show files that would be deleted without deleting them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    target_dirs = ["configs/experiments", "data", "results"]

    files_to_delete = collect_files(root_dir, target_dirs)

    if not files_to_delete:
        print("No files found in target directories.")
        return 0

    print(f"Files selected for deletion ({len(files_to_delete)}):")
    for path in files_to_delete:
        print(f"  {path.relative_to(root_dir)}")

    if args.dry_run:
        print("\nDry-run completed. No files were deleted.")
        return 0

    if not args.force:
        reply = input("\nProceed with deletion? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            return 0

    for path in files_to_delete:
        path.unlink(missing_ok=True)

    print(f"Done. Deleted {len(files_to_delete)} files. Directory structure preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
