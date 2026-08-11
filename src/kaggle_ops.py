"""Kaggle run-state, artifact sealing, and Google Drive backup helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any, length: int | None = None) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return digest[:length] if length else digest


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class RagRunConfig:
    project: str = "mission14"
    stage: str = "kaggle_rag_v1"
    source_revision: str = "unspecified"
    seed: int = 42
    embedding_model: str = "nlpai-lab/KURE-v1"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    generator_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    dense_k: int = 15
    bm25_k: int = 15
    rrf_top_n: int = 12
    final_top_n: int = 5
    primary_metric: str = "all_facts_covered@5"
    device_policy: str = "CUDA required for reranker and generator"
    allow_cpu_fallback: bool = False
    official_test_policy: str = "No hidden official test; gold retrieval set is sealed before index selection"


def create_run_layout(
    root: str | Path,
    config: RagRunConfig,
    resume_run_id: str = "",
) -> dict[str, Any]:
    config_dict = asdict(config)
    config_hash = canonical_hash(config_dict, 12)
    run_id = resume_run_id or f"{config.stage}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{config_hash}"
    run_dir = Path(root) / config.project / run_id
    paths = {
        "run_dir": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "indexes": run_dir / "indexes",
        "results": run_dir / "results",
        "logs": run_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "config.json", config_dict)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != config_hash:
            raise RuntimeError("Resume denied: config hash differs from the saved run.")
    else:
        manifest = {
            "run_id": run_id,
            "config_hash": config_hash,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "running",
            "completed_phases": [],
            "input_pdf": None,
            "device": None,
            "drive_backups": [],
            "warnings": [],
        }
        atomic_write_json(manifest_path, manifest)
    return {
        "run_id": run_id,
        "config_hash": config_hash,
        "manifest": manifest,
        **paths,
    }


def update_manifest(run_dir: str | Path, **updates: Any) -> dict[str, Any]:
    path = Path(run_dir) / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    manifest["updated_at"] = utc_now()
    atomic_write_json(path, manifest)
    return manifest


def mark_phase_complete(
    run_dir: str | Path,
    phase: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(run_dir) / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    phases = list(manifest.get("completed_phases", []))
    phases = [item for item in phases if item.get("phase") != phase]
    phases.append({"phase": phase, "completed_at": utc_now(), "evidence": dict(evidence or {})})
    manifest["completed_phases"] = phases
    manifest["updated_at"] = utc_now()
    atomic_write_json(path, manifest)
    return manifest


def bind_input_pdf(run_dir: str | Path, pdf_info: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(run_dir) / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    previous = manifest.get("input_pdf")
    current = {
        "sha256": str(pdf_info["sha256"]),
        "bytes": int(pdf_info["bytes"]),
        "page_count": int(pdf_info["page_count"]),
    }
    if previous and previous != current:
        raise RuntimeError("Resume denied: PDF fingerprint differs from the saved run.")
    manifest["input_pdf"] = current
    manifest["updated_at"] = utc_now()
    atomic_write_json(path, manifest)
    return manifest


def archive_run(run_dir: str | Path, destination_dir: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    destination_dir = Path(destination_dir or run_dir.parent)
    destination_dir.mkdir(parents=True, exist_ok=True)
    archive_base = destination_dir / f"{run_dir.name}__checkpoint"
    archive = Path(shutil.make_archive(str(archive_base), "zip", root_dir=run_dir))
    if not archive.exists() or archive.stat().st_size == 0:
        raise RuntimeError("Checkpoint archive creation failed.")
    return archive


def create_drive_service_from_kaggle_secrets() -> tuple[Any, str]:
    """Build a Drive v3 client without printing or persisting OAuth secrets."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    token_info = json.loads(secrets.get_secret("GOOGLE_OAUTH_TOKEN_JSON"))
    folder_id = secrets.get_secret("GOOGLE_DRIVE_BACKUP_FOLDER_ID")
    scopes = ["https://www.googleapis.com/auth/drive.file"]
    credentials = Credentials.from_authorized_user_info(token_info, scopes)
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials.valid:
        raise RuntimeError("Google Drive OAuth credential is invalid or cannot refresh.")
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    return service, folder_id


def drive_upload_verified(
    service: Any,
    local_path: str | Path,
    folder_id: str,
    file_id: str | None = None,
) -> dict[str, Any]:
    """Create/update a Drive file and verify size/checksum from returned metadata."""
    import mimetypes

    from googleapiclient.http import MediaFileUpload

    local_path = Path(local_path)
    mime_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
    fields = "id,name,modifiedTime,size,md5Checksum,parents"
    if file_id:
        remote = (
            service.files()
            .update(fileId=file_id, media_body=media, fields=fields)
            .execute()
        )
    else:
        remote = (
            service.files()
            .create(
                body={"name": local_path.name, "parents": [folder_id]},
                media_body=media,
                fields=fields,
            )
            .execute()
        )
    if int(remote.get("size", -1)) != local_path.stat().st_size:
        raise RuntimeError("Drive backup size verification failed.")
    result = {
        "file_id": remote["id"],
        "name": remote.get("name"),
        "size": int(remote["size"]),
        "md5": remote.get("md5Checksum"),
        "local_sha256": sha256_file(local_path),
        "modified_time": remote.get("modifiedTime"),
        "uploaded_at": utc_now(),
    }
    return result


def restore_run_from_drive(file_id: str, destination_run_dir: str | Path) -> Path:
    """Restore a checkpoint archive after validating every ZIP extraction target."""
    import io
    import zipfile

    from googleapiclient.http import MediaIoBaseDownload

    service, _ = create_drive_service_from_kaggle_secrets()
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    destination = Path(destination_run_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe path in Drive checkpoint archive: {member.filename}")
        archive.extractall(destination)
    manifest_path = destination / "run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Restored archive has no run_manifest.json.")
    json.loads(manifest_path.read_text(encoding="utf-8"))
    return destination


def backup_run_to_drive(run_dir: str | Path) -> dict[str, Any]:
    """Best-effort Drive backup; callers record failures and keep Kaggle artifacts."""
    run_dir = Path(run_dir)
    archive = archive_run(run_dir)
    service, folder_id = create_drive_service_from_kaggle_secrets()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    backups = list(manifest.get("drive_backups", []))
    existing_file_id = backups[-1].get("file_id") if backups else None
    result = drive_upload_verified(
        service, archive, folder_id, file_id=existing_file_id
    )
    backups.append(result)
    update_manifest(run_dir, drive_backups=backups)
    return result


def seal_run(run_dir: str | Path, metrics_path: str | Path) -> dict[str, Any]:
    """Create COMPLETE.lock.json last, after core artifacts can be re-read."""
    run_dir = Path(run_dir)
    metrics_path = Path(metrics_path)
    manifest_path = run_dir / "run_manifest.json"
    if not metrics_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("metrics.json and run_manifest.json are required before sealing.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock = {
        "run_id": manifest["run_id"],
        "config_hash": manifest["config_hash"],
        "sealed_at": utc_now(),
        "metrics_sha256": sha256_file(metrics_path),
        "manifest_sha256_before_seal": sha256_file(manifest_path),
        "sealed_test": bool(metrics.get("sealed_test", False)),
        "post_test_tuning_allowed": bool(metrics.get("post_test_tuning_allowed", True)),
    }
    atomic_write_json(run_dir / "COMPLETE.lock.json", lock)
    update_manifest(run_dir, status="complete")
    json.loads((run_dir / "COMPLETE.lock.json").read_text(encoding="utf-8"))
    return lock
