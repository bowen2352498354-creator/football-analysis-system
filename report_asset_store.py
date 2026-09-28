"""Persistent image assets for archived coach reports.

The training archive used to embed the full impact-frame image as Base64 in
``global_training_db.json``.  This module validates that image, writes it to a
record-scoped directory atomically, and returns JSON-safe metadata.  It is
deliberately independent from FastAPI so it can be reused by migrations and
tested without starting the application.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from PIL import Image, UnidentifiedImageError


SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_ASSET_ROOT = SCRIPT_DIR / "report-assets"

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000

_DATA_URI_RE = re.compile(
    r"^data:(?P<mime>image/[a-zA-Z0-9.+-]+);base64,(?P<data>.*)$",
    re.DOTALL,
)
_SAFE_RECORD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_FORMAT_META = {
    "JPEG": ("jpg", "image/jpeg"),
    "PNG": ("png", "image/png"),
    "WEBP": ("webp", "image/webp"),
}


class AssetStorageError(ValueError):
    """Raised when an image payload cannot be safely persisted."""


def _validate_record_id(record_id: str) -> str:
    clean = str(record_id or "").strip()
    if not _SAFE_RECORD_ID_RE.fullmatch(clean):
        raise AssetStorageError("invalid record id for report asset")
    return clean


def decode_image_payload(payload: Optional[str]) -> Optional[dict[str, Any]]:
    """Decode and validate a Base64/data-URI image.

    Returns the original encoded bytes plus canonical metadata.  ``None`` or
    blank input means that the attempt has no archived impact frame.
    """

    if payload is None:
        return None
    text = str(payload).strip()
    if not text:
        return None

    declared_mime = ""
    encoded = text
    match = _DATA_URI_RE.match(text)
    if match:
        declared_mime = match.group("mime").lower()
        encoded = match.group("data")
    elif text.lower().startswith("data:"):
        raise AssetStorageError("impact frame must be a Base64 image data URI")

    # Browser encoders may insert whitespace/newlines into long Base64 values.
    encoded = "".join(encoded.split())
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AssetStorageError("impact frame contains invalid Base64 data") from exc

    if not raw:
        raise AssetStorageError("impact frame is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise AssetStorageError(
            f"impact frame exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
        )

    try:
        with Image.open(io.BytesIO(raw)) as image:
            image_format = str(image.format or "").upper()
            width, height = image.size
            image.verify()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise AssetStorageError("impact frame is not a readable image") from exc

    if image_format not in _FORMAT_META:
        raise AssetStorageError(f"unsupported impact frame format: {image_format or 'unknown'}")
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise AssetStorageError("impact frame dimensions are outside the safe range")

    extension, actual_mime = _FORMAT_META[image_format]
    declared_alias = "image/jpeg" if declared_mime == "image/jpg" else declared_mime
    if declared_alias and declared_alias != actual_mime:
        raise AssetStorageError("impact frame MIME type does not match its content")

    return {
        "bytes": raw,
        "extension": extension,
        "mimeType": actual_mime,
        "width": int(width),
        "height": int(height),
        "sizeBytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def save_impact_frame(
    record_id: str,
    payload: Optional[str],
    *,
    root: Optional[os.PathLike[str] | str] = None,
) -> Optional[dict[str, Any]]:
    """Persist one validated impact frame and return archive metadata.

    The write is atomic: a temporary file is created in the target directory
    and then replaced into its final name.  Existing assets for the same record
    can therefore be retried safely without leaving a partial image behind.
    """

    decoded = decode_image_payload(payload)
    if decoded is None:
        return None

    clean_id = _validate_record_id(record_id)
    asset_root = Path(root) if root is not None else REPORT_ASSET_ROOT
    asset_root = asset_root.resolve()
    record_dir = asset_root / "records" / clean_id
    record_dir.mkdir(parents=True, exist_ok=True)

    extension = str(decoded["extension"])
    target = record_dir / f"impact.{extension}"
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=".impact-",
        suffix=f".{extension}.tmp",
        dir=str(record_dir),
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(decoded["bytes"])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

    relative_path = target.relative_to(asset_root).as_posix()
    return {
        "path": relative_path,
        "mimeType": decoded["mimeType"],
        "sha256": decoded["sha256"],
        "width": decoded["width"],
        "height": decoded["height"],
        "sizeBytes": decoded["sizeBytes"],
    }


def resolve_asset_path(
    asset: Any,
    *,
    root: Optional[os.PathLike[str] | str] = None,
) -> Optional[Path]:
    """Resolve stored metadata without allowing paths outside the asset root."""

    if not isinstance(asset, dict):
        return None
    relative = str(asset.get("path") or "").strip()
    if not relative:
        return None

    asset_root = (Path(root) if root is not None else REPORT_ASSET_ROOT).resolve()
    candidate = (asset_root / Path(relative)).resolve()
    try:
        candidate.relative_to(asset_root)
    except ValueError as exc:
        raise AssetStorageError("report asset path escapes the configured root") from exc
    return candidate
