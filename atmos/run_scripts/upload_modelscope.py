#!/usr/bin/env python3
"""Upload the ATMOS model and dataset files listed in packages.json to ModelScope."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("atmos.upload")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def checked_file(root: Path, relative: str) -> Path:
    """Require a literal, nonempty file path within the asset root."""
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Invalid manifest path: {relative!r}")
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != relative
        or any(char in relative for char in "*?[]\\\n\r")
    ):
        raise ValueError(f"Expected a literal relative file path: {relative!r}")
    local = root / relative
    if not local.resolve().is_relative_to(root):
        raise ValueError(f"File points outside the asset root: {relative}")
    if not local.is_file() or local.stat().st_size == 0:
        raise ValueError(f"Missing or empty asset: {local}")
    return local


def upload_files(root: Path, manifest: dict, repo_type: str) -> list[str]:
    """Validate the manifest entries for one repository and return exact paths."""
    if repo_type == "model":
        files = list(manifest["model_files"])
        for name in files:
            if not name.startswith("atmos_checkpoints/") or not name.endswith((".pt", ".yaml")):
                raise ValueError(f"Unexpected model file: {name}")
    else:
        files = []
        for archive in manifest["archives"]:
            name = archive["archive"]
            if not name.startswith(("mdCATH_tempo/", "misato/")) or not name.endswith((".tar", ".tar.gz")):
                raise ValueError(f"Unexpected data archive: {name}")
            local = checked_file(root, name)
            if local.stat().st_size != archive["archive_bytes"]:
                raise ValueError(f"Archive size differs from packages.json: {name}")
            files.append(name)
        for name in manifest["data_metadata_files"]:
            if name.startswith("atmos_checkpoints/") or not name.endswith((".csv", ".json", ".txt")):
                raise ValueError(f"Unexpected dataset metadata: {name}")
            files.append(name)
        files.append("packages.json")
    if not files or len(files) != len(set(files)):
        raise ValueError(f"Empty or duplicated {repo_type} file list")
    for name in files:
        checked_file(root, name)
    return sorted(files)


def report_progress(event: dict) -> None:
    kind = event.get("event")
    if kind in {"batch_committed", "upload_progress"}:
        LOGGER.info(
            "%s: %s; committed=%s, cached=%s, total=%s; transferred=%.2f GiB",
            event["repo_id"], kind, event["committed_files"],
            event["skipped_files"], event["total_files"],
            event.get("uploaded_bytes", 0) / 1024**3,
        )
    elif kind == "batch_failed":
        LOGGER.warning("%s: batch failed: %s", event["repo_id"], event.get("error"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Upload ATMOS assets to the model and dataset repositories on ModelScope.",
        epilog=(
            "Requires modelscope-hub>=0.5.0 and an account authenticated with ms-hub login. "
            "Uses exact paths from packages.json and preserves their directory structure. "
            "Rerun with the same cache directory to skip unchanged files already committed. "
            "Logs go to stdout/stderr; redirect both when running in the background."
        ),
    )
    parser.add_argument(
        "--asset-root", type=Path,
        default=Path(os.environ.get("ATMOS_ROOT_DIR", REPO_ROOT / "atmos_oss")),
        help="Asset directory containing packages.json (default: ATMOS_ROOT_DIR or repository atmos_oss/).",
    )
    parser.add_argument("--only", choices=("all", "model", "dataset"), default="all",
                        help="Repositories to upload (default: all, model first).")
    parser.add_argument("--model-repo", default="SmartVector/ATMOS")
    parser.add_argument("--dataset-repo", default="SmartVector/ATMOS-dataset")
    parser.add_argument("--revision", default="master", help="Destination branch (default: master).")
    parser.add_argument("--max-workers", type=positive_int, default=4,
                        help="Concurrent file transfers (default: 4).")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "output/modelscope_upload",
                        help="Persistent upload cache (default: repository output/modelscope_upload/).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and list local files without network access or uploading.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = args.asset_root.expanduser().resolve()
    manifest = json.loads(checked_file(root, "packages.json").read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported packages.json format_version; expected 1")
    targets = []
    for repo_type, repo_id in (("model", args.model_repo), ("dataset", args.dataset_repo)):
        if args.only not in ("all", repo_type):
            continue
        if len(repo_id.split("/")) != 2 or any(not part for part in repo_id.split("/")):
            raise ValueError(f"Expected owner/name repository ID: {repo_id!r}")
        files = upload_files(root, manifest, repo_type)
        size = sum((root / name).stat().st_size for name in files)
        LOGGER.info("%s (%s): %d files, %.3f GiB", repo_id, repo_type, len(files), size / 1024**3)
        for name in files:
            LOGGER.info("  %s -> %s:%s", name, repo_id, name)
        targets.append((repo_type, repo_id, files))
    if args.dry_run:
        LOGGER.info("Dry run complete; no network requests or uploads performed.")
        return 0

    try:
        from modelscope_hub import HubApi
    except ImportError as exc:
        raise RuntimeError("Install the upload dependency: python -m pip install 'modelscope-hub>=0.5.0'") from exc

    api = HubApi(endpoint="https://modelscope.cn")
    api.whoami()
    for repo_type, repo_id, _ in targets:
        api.get_repo(repo_id, repo_type=repo_type)
        LOGGER.info("Repository accessible: %s (%s)", repo_id, repo_type)

    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for repo_type, repo_id, files in targets:
        cache_name = f"{repo_type}-{quote(repo_id, safe='')}-{quote(args.revision, safe='')}.json"
        LOGGER.info("Uploading %s; resume cache: %s", repo_id, cache_dir / cache_name)
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=root,
            path_in_repo="",
            revision=args.revision,
            commit_message="Upload ATMOS model assets" if repo_type == "model" else "Upload ATMOS datasets",
            allow_patterns=files,
            max_workers=args.max_workers,
            use_cache=True,
            tracker_path=cache_dir / cache_name,
            disable_tqdm=True,
            sync_remote_repo=False,
            progress_callback=report_progress,
        )
        LOGGER.info("Completed: %s (%s)", repo_id, repo_type)
    LOGGER.info("All requested uploads completed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOGGER.error("Interrupted. Rerun with the same cache directory to continue.")
        sys.exit(130)
    except Exception as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        LOGGER.error("Upload incomplete. Fix the error and rerun with the same cache directory.")
        sys.exit(1)
