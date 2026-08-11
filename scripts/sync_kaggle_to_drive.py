#!/usr/bin/env python3
"""Pull the latest Kaggle notebook/output and replace an existing Drive notebook.

This is the post-run half of the Mission 14 sync flow. The Kaggle notebook backs
up run artifacts during execution; this script preserves the existing Google
Drive notebook file ID after Kaggle saves a new Version.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


START = "<!-- AUTO_SUMMARY_START -->"
END = "<!-- AUTO_SUMMARY_END -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True, help="Kaggle owner/notebook-slug")
    parser.add_argument("--drive-token-json", required=True, type=Path)
    parser.add_argument("--drive-notebook-file-id", required=True)
    parser.add_argument("--drive-results-folder-id")
    parser.add_argument("--backup-dir", type=Path, default=Path("drive_sync_backups"))
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def latest_matching(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def summary_from_metrics(metrics: dict) -> str:
    retrieval = metrics.get("retrieval", {})
    winner = metrics.get("best_retriever", "미확정")
    winner_metrics = retrieval.get(winner, {})
    return f"""# Mission 14 Kaggle 실행 결과

## 1. 최종 결과
- Run ID: `{metrics.get('run_id', 'unknown')}`
- Best retriever: `{winner}`
- All facts covered@5: `{winner_metrics.get('all_facts_covered@5', 'n/a')}`
- MRR: `{winner_metrics.get('mrr', 'n/a')}`

## 2. 실험 흐름 및 Notebook 구성
PDF 검증 → gold evidence gate → C1/C2/C3 → Dense → Hybrid → Rerank → Qwen 비교

## 3. 전체 실험 결과 요약
상세 수치는 Kaggle output의 `metrics.json`과 `stage_summary.md`를 기준으로 합니다.

## 4. 최종 결론
검색 성능과 생성 답변을 분리 평가했으며, 관련 페이지 하나와 필수 사실 전체 커버를 구분했습니다.

## 5. 전체 코드 및 실행 결과
Kaggle에서 저장된 최신 Version입니다."""


def update_first_summary(notebook: dict, summary: str) -> None:
    block = f"{START}\n{summary.strip()}\n{END}\n"
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown" and START in source:
            before = source.split(START, 1)[0]
            after = source.split(END, 1)[1] if END in source else ""
            cell["source"] = (before + block + after).splitlines(keepends=True)
            return
    notebook.setdefault("cells", []).insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": block.splitlines(keepends=True),
        },
    )


def drive_service(token_path: Path):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    scopes = ["https://www.googleapis.com/auth/drive.file"]
    info = json.loads(token_path.read_text(encoding="utf-8"))
    credentials = Credentials.from_authorized_user_info(info, scopes)
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials.valid:
        raise RuntimeError("Drive OAuth token is invalid or cannot refresh.")
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def update_drive_notebook(service, file_id: str, notebook_path: Path) -> dict:
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(
        str(notebook_path), mimetype="application/x-ipynb+json", resumable=True
    )
    return (
        service.files()
        .update(
            fileId=file_id,
            media_body=media,
            fields="id,name,size,md5Checksum,modifiedTime",
        )
        .execute()
    )


def download_drive_file(service, file_id: str) -> bytes:
    from googleapiclient.http import MediaIoBaseDownload

    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def upload_results_archive(service, folder_id: str, output_dir: Path) -> dict:
    from googleapiclient.http import MediaFileUpload

    archive_base = output_dir.parent / "mission14_kaggle_output"
    archive = Path(shutil.make_archive(str(archive_base), "zip", root_dir=output_dir))
    media = MediaFileUpload(str(archive), mimetype="application/zip", resumable=True)
    return (
        service.files()
        .create(
            body={"name": archive.name, "parents": [folder_id]},
            media_body=media,
            fields="id,name,size,md5Checksum,modifiedTime",
        )
        .execute()
    )


def main() -> None:
    args = parse_args()
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mission14_kaggle_sync_") as temp:
        root = Path(temp)
        source_dir = root / "source"
        output_dir = root / "output"
        source_dir.mkdir()
        output_dir.mkdir()
        run_checked(["kaggle", "kernels", "pull", args.kernel, "-p", str(source_dir), "-m"])
        run_checked(["kaggle", "kernels", "output", args.kernel, "-p", str(output_dir), "-o"])

        notebook_path = latest_matching(source_dir, "*.ipynb")
        if notebook_path is None:
            raise FileNotFoundError("Kaggle source pull did not contain an .ipynb file.")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        metrics_path = latest_matching(output_dir, "metrics.json")
        if metrics_path:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            update_first_summary(notebook, summary_from_metrics(metrics))
        notebook_path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        local_backup = args.backup_dir / f"{notebook_path.stem}__before_drive_update__{timestamp}.ipynb"
        shutil.copy2(notebook_path, local_backup)

        service = drive_service(args.drive_token_json)
        remote = update_drive_notebook(
            service, args.drive_notebook_file_id, notebook_path
        )
        downloaded = download_drive_file(service, args.drive_notebook_file_id)
        json.loads(downloaded.decode("utf-8"))
        local_bytes = notebook_path.read_bytes()
        if sha256_bytes(downloaded) != sha256_bytes(local_bytes):
            raise RuntimeError("Drive notebook readback checksum differs from local bytes.")
        if remote["id"] != args.drive_notebook_file_id:
            raise RuntimeError("Drive notebook file ID changed unexpectedly.")

        result_remote = None
        if args.drive_results_folder_id:
            result_remote = upload_results_archive(
                service, args.drive_results_folder_id, output_dir
            )

        print(
            json.dumps(
                {
                    "kernel": args.kernel,
                    "drive_notebook_file_id": remote["id"],
                    "drive_notebook_size": int(remote["size"]),
                    "drive_output_file_id": result_remote["id"] if result_remote else None,
                    "local_backup": str(local_backup),
                    "notebook_json_parse": "passed",
                    "readback_checksum": "passed",
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
