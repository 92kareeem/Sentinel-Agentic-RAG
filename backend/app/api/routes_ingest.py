"""POST /v1/documents (presigned upload) + /v1/documents/{doc_id}/index.

Role in architecture: the presigned URL means file bytes go browser -> S3
directly; Lambda never proxies them (Lambda bills by duration and caps
payloads at 6 MB — proxying uploads would be slow, capped, and expensive).
The index endpoint then pulls from S3, chunks, embeds, and merges.
"""

import re
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.guardrails.auth import resolve_user
from app.models.schemas import PresignedUploadResponse

router = APIRouter()

_UPLOAD_EXPIRY_S = 900


@router.post("/documents")
def request_upload(
    filename: str, user: dict[str, Any] = Depends(resolve_user)
) -> PresignedUploadResponse:
    settings = get_settings()
    if settings.local_mode:
        raise HTTPException(status_code=501, detail="uploads require AWS mode (P4)")
    if not re.fullmatch(r"[\w.\- ]{1,128}\.(pdf|md|txt)", filename, re.I):
        raise HTTPException(status_code=400, detail="filename must be .pdf/.md/.txt")

    import boto3

    doc_id = uuid.uuid4().hex[:12]
    key = f"uploads/{user['user_id']}/{doc_id}/{filename}"
    s3 = boto3.client("s3", region_name=settings.aws_region)
    url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": settings.s3_bucket_docs, "Key": key},
        ExpiresIn=_UPLOAD_EXPIRY_S,
    )
    return PresignedUploadResponse(
        doc_id=doc_id, upload_url=url, expires_in_seconds=_UPLOAD_EXPIRY_S
    )


@router.post("/documents/{doc_id}/index")
def index_document(doc_id: str, user: dict[str, Any] = Depends(resolve_user)) -> dict:
    settings = get_settings()
    if settings.local_mode:
        raise HTTPException(status_code=501, detail="uploads require AWS mode (P4)")
    # P4: download from uploads/{user_id}/{doc_id}/, chunk, embed, merge into
    # the index artifacts, upload to /index/, bump index version.
    raise HTTPException(status_code=501, detail="index merge lands in P4")
