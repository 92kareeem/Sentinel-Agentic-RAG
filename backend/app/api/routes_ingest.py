"""POST /v1/documents (presigned upload) + /v1/documents/{doc_id}/index.

Role in architecture: the presigned POST means file bytes go browser -> S3
directly; Lambda never proxies them (Lambda bills by duration and caps
payloads at 6 MB — proxying uploads would be slow, capped, and expensive).
A presigned POST (not PUT) is used specifically so a content-length-range
condition can cap the upload at 5 MB *at S3*, before a byte is stored. The
index endpoint then pulls the object from S3, chunks, embeds, and merges it
into the live index.
"""

import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.agents import retriever
from app.config import get_settings
from app.guardrails.auth import resolve_user
from app.guardrails.quota import check_upload_quota, record_upload
from app.models.schemas import IndexJobResponse, PresignedUploadResponse

router = APIRouter()

_UPLOAD_EXPIRY_S = 900
_FILENAME_RE = re.compile(r"[\w.\- ]{1,128}\.(pdf|md|txt)", re.I)


def _require_aws() -> Any:
    settings = get_settings()
    if settings.local_mode:
        raise HTTPException(status_code=501, detail="uploads require the deployed (AWS) API")
    return settings


def _safe_filename(filename: str) -> str:
    if not _FILENAME_RE.fullmatch(filename):
        raise HTTPException(status_code=400, detail="filename must be .pdf/.md/.txt")
    return filename


@router.post("/documents")
def request_upload(
    filename: str, user: dict[str, Any] = Depends(resolve_user)
) -> PresignedUploadResponse:
    settings = _require_aws()
    _safe_filename(filename)
    check_upload_quota(user, incoming_bytes=0)  # doc-count check before issuing a URL

    import boto3

    doc_id = uuid.uuid4().hex[:12]
    key = f"uploads/{user['user_id']}/{doc_id}/{filename}"
    s3 = boto3.client("s3", region_name=settings.aws_region)
    post = s3.generate_presigned_post(
        Bucket=settings.s3_bucket_docs,
        Key=key,
        Conditions=[["content-length-range", 1, settings.max_upload_bytes]],
        ExpiresIn=_UPLOAD_EXPIRY_S,
    )
    return PresignedUploadResponse(
        doc_id=doc_id,
        upload_url=post["url"],
        fields=post["fields"],
        filename=filename,
        max_bytes=settings.max_upload_bytes,
        expires_in_seconds=_UPLOAD_EXPIRY_S,
    )


@router.post("/documents/{doc_id}/index")
def index_document(
    doc_id: str, filename: str, user: dict[str, Any] = Depends(resolve_user)
) -> IndexJobResponse:
    """Pull the just-uploaded object and merge it into the live index.

    filename is passed by the client so the exact S3 key can be rebuilt —
    avoids granting the Lambda role s3:ListBucket just to discover it.
    """
    settings = _require_aws()
    _safe_filename(filename)

    import boto3
    from botocore.exceptions import ClientError

    from app.rag.ingest_runtime import merge_document

    key = f"uploads/{user['user_id']}/{doc_id}/{filename}"
    local = Path("/tmp") / f"{doc_id}_{filename}"
    s3 = boto3.client("s3", region_name=settings.aws_region)
    try:
        s3.download_file(settings.s3_bucket_docs, key, str(local))
    except ClientError as exc:
        raise HTTPException(status_code=404, detail="uploaded object not found") from exc

    size = local.stat().st_size
    if size > settings.max_upload_bytes:  # defense in depth vs the S3 condition
        local.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="document exceeds size limit")
    check_upload_quota(user, incoming_bytes=size)

    try:
        chunks_indexed, version = merge_document(local)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        local.unlink(missing_ok=True)

    retriever.reset_cache()  # so the next query sees the new document
    record_upload(user, size)
    return IndexJobResponse(doc_id=doc_id, chunks_indexed=chunks_indexed, index_version=version)
