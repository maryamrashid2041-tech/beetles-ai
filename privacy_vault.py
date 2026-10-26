# privacy_vault.py
"""
Privacy Vault for AI Forensics Platform

Purpose:
  - Encrypt/decrypt sensitive data (media hashes, metadata, report logs) before storage.
  - Redact reports for safe sharing by removing filenames, IP addresses, and identifiers.

Dependencies:
  - cryptography

Install:
  pip install cryptography

Security notes:
  - Keep your Fernet key secret. Store it in a secrets manager or environment variable.
  - Fernet provides authenticated encryption (confidentiality + integrity).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple, Union

try:
    from cryptography.fernet import Fernet, InvalidToken  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "cryptography is required. Install with: pip install cryptography"
    ) from e


JsonLike = Union[Dict[str, Any], list, str, int, float, bool, None]


# -----------------------------
# Redaction helpers
# -----------------------------

_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)
# A conservative IPv6 match (not perfect, but useful for redaction).
_IPV6_RE = re.compile(r"\b(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}\b")
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
_PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}\b"
)
# Rough identifiers: long tokens / secrets / IDs
_LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_\-]{24,}\b")

# Filename/path patterns
_PATH_SEGMENT_RE = re.compile(r"([A-Za-z]:\\[^ \n\r\t<>|\"']+)|(/[^ \n\r\t<>|\"']+)")
_FILENAME_RE = re.compile(
    r"\b[\w,\s\-.]+\.(?:png|jpg|jpeg|webp|gif|bmp|tiff|wav|mp3|m4a|aac|flac|ogg|mp4|mov|mkv|avi)\b",
    re.IGNORECASE,
)

# Keys that often carry sensitive identifiers
_SENSITIVE_KEYS = {
    "filename",
    "file_name",
    "filepath",
    "file_path",
    "path",
    "local_path",
    "remote_path",
    "ip",
    "ip_address",
    "client_ip",
    "user_ip",
    "email",
    "phone",
    "user",
    "username",
    "full_name",
    "name",
    "first_name",
    "last_name",
    "address",
    "device_id",
    "session_id",
    "cookie",
    "auth",
    "authorization",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "secret",
    "ssn",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def _safe_json_dumps(obj: Any) -> str:
    """
    Deterministic JSON serialization for encryption/storage.
    """
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)
    except Exception as e:
        raise ValueError(f"Object is not JSON-serializable: {e}") from e


def _safe_json_loads(s: Union[str, bytes]) -> Any:
    try:
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        return json.loads(s)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}") from e


def _redact_text(text: str) -> str:
    """
    Redact common personal/sensitive patterns from free text.
    """
    if not isinstance(text, str):
        return text  # type: ignore[return-value]

    t = text

    # Remove explicit paths and filenames.
    t = _PATH_SEGMENT_RE.sub("[REDACTED_PATH]", t)
    t = _FILENAME_RE.sub("[REDACTED_FILENAME]", t)

    # Remove network / contact info.
    t = _IPV4_RE.sub("[REDACTED_IP]", t)
    t = _IPV6_RE.sub("[REDACTED_IP]", t)
    t = _EMAIL_RE.sub("[REDACTED_EMAIL]", t)

    # Phone numbers can be noisy; redact conservatively.
    t = _PHONE_RE.sub("[REDACTED_PHONE]", t)

    # Long tokens (session IDs, JWT fragments, API keys).
    t = _LONG_TOKEN_RE.sub("[REDACTED_TOKEN]", t)

    return t


def _redact_key_value(key: str, value: Any) -> Tuple[str, Any]:
    """
    Redact a single key/value pair with sensitivity heuristics.
    """
    k_norm = str(key).strip().lower()

    # Direct key-based redaction
    if k_norm in _SENSITIVE_KEYS or any(s in k_norm for s in ["password", "passwd", "pwd", "secret", "token", "apikey", "api_key"]):
        return key, "[REDACTED]"

    # Indirect key-based redaction: keys that suggest file or network identifiers
    if any(s in k_norm for s in ["file", "path", "ip", "email", "phone", "user", "name", "address", "device", "session"]):
        # For some keys we may still keep non-sensitive primitive values;
        # but safest is to redact or sanitize.
        if isinstance(value, str):
            return key, _redact_text(value)
        return key, "[REDACTED]"

    # Otherwise, sanitize textual content for embedded PII
    if isinstance(value, str):
        return key, _redact_text(value)

    return key, value


def _recursive_redact(obj: Any, max_depth: int = 12, _depth: int = 0) -> Any:
    if _depth > max_depth:
        return "[REDACTED_DEPTH_LIMIT]"

    if obj is None or isinstance(obj, (int, float, bool)):
        return obj

    if isinstance(obj, str):
        return _redact_text(obj)

    if isinstance(obj, (bytes, bytearray, memoryview)):
        # Never include raw bytes in shareable reports
        return "[REDACTED_BYTES]"

    if isinstance(obj, list):
        return [_recursive_redact(x, max_depth=max_depth, _depth=_depth + 1) for x in obj]

    if _is_mapping(obj):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            try:
                rk, rv = _redact_key_value(str(k), v)
                out[rk] = _recursive_redact(rv, max_depth=max_depth, _depth=_depth + 1)
            except Exception:
                # Fail closed: redact problematic entries.
                out[str(k)] = "[REDACTED]"
        return out

    # Unknown types -> stringified then redacted
    return _redact_text(str(obj))


def redact_report(report_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Strip raw filenames, IP addresses, and personal identifiers from a report dict
    before generating a shareable report.

    This function:
      - Redacts sensitive keys (filename, ip, email, tokens, etc.)
      - Scrubs PII patterns from nested text fields (emails, IPs, paths, tokens)
      - Removes raw binary blobs if present

    Args:
      report_dict: original report dictionary

    Returns:
      A deep-redacted copy safe to share.
    """
    if report_dict is None:
        raise ValueError("report_dict must not be None.")
    if not _is_mapping(report_dict):
        raise TypeError(f"report_dict must be a mapping/dict-like object, got {type(report_dict).__name__}.")

    redacted = _recursive_redact(dict(report_dict))
    # Attach metadata for traceability (does not include sensitive data)
    if isinstance(redacted, dict):
        redacted.setdefault("_redaction", {})
        if isinstance(redacted["_redaction"], dict):
            redacted["_redaction"].update(
                {
                    "redacted_at": _utc_now_iso(),
                    "policy": "strip_filenames_ips_identifiers_v1",
                }
            )
    return redacted  # type: ignore[return-value]


# -----------------------------
# PrivacyVault
# -----------------------------

@dataclass(frozen=True)
class EncryptedBlob:
    """
    A structured wrapper for encrypted payloads.
    token is a urlsafe-base64 Fernet token (bytes).
    """
    token: bytes
    created_at: str
    algorithm: str = "Fernet"


class PrivacyVault:
    """
    Encrypt/decrypt sensitive user media hashes, metadata, and report logs before storing.

    Recommended use:
      vault = PrivacyVault.from_env("FORENSICS_FERNET_KEY")  # or PrivacyVault.generate_key()
      encrypted = vault.encrypt_dict({"media_sha256": "...", "scores": {...}})
      decrypted = vault.decrypt_dict(encrypted)

    Key handling:
      - Provide a Fernet key (urlsafe base64-encoded 32-byte key).
      - Keep it secret and rotate periodically.
    """

    def __init__(self, fernet_key: Union[str, bytes]):
        if fernet_key is None:
            raise ValueError("fernet_key must not be None.")
        if isinstance(fernet_key, str):
            fernet_key_b = fernet_key.encode("utf-8")
        elif isinstance(fernet_key, (bytes, bytearray, memoryview)):
            fernet_key_b = bytes(fernet_key)
        else:
            raise TypeError(f"fernet_key must be str or bytes, got {type(fernet_key).__name__}.")

        # Validate key format early.
        try:
            # Fernet expects a urlsafe-base64-encoded 32-byte key (44 chars).
            # We'll attempt a decode to confirm it's plausible; Fernet() will also validate.
            _ = base64.urlsafe_b64decode(fernet_key_b)
            self._fernet = Fernet(fernet_key_b)
        except Exception as e:
            raise ValueError(f"Invalid Fernet key: {e}") from e

    # ---------
    # Key utilities
    # ---------

    @staticmethod
    def generate_key() -> str:
        """
        Generate a new Fernet key (string).
        """
        try:
            return Fernet.generate_key().decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to generate Fernet key: {e}") from e

    @classmethod
    def from_env(cls, env_var: str = "FORENSICS_FERNET_KEY") -> "PrivacyVault":
        """
        Instantiate from an environment variable.
        """
        if not isinstance(env_var, str) or not env_var.strip():
            raise ValueError("env_var must be a non-empty string.")
        key = os.getenv(env_var)
        if not key:
            raise RuntimeError(
                f"Environment variable {env_var!r} is not set. "
                f"Set it to a Fernet key (use PrivacyVault.generate_key())."
            )
        return cls(key)

    # ---------
    # Hashing utilities (non-reversible, for indexing/dedup)
    # ---------

    @staticmethod
    def sha256_hex(data: Union[bytes, bytearray, memoryview]) -> str:
        """
        Compute SHA-256 hex digest of data (e.g., media bytes).
        Useful for media fingerprinting without storing the raw media.
        """
        if data is None:
            raise ValueError("data must not be None.")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"data must be bytes-like, got {type(data).__name__}.")
        b = bytes(data)
        if not b:
            raise ValueError("data is empty.")
        return hashlib.sha256(b).hexdigest()

    # ---------
    # Encryption/decryption for bytes
    # ---------

    def encrypt_bytes(self, plaintext: Union[bytes, bytearray, memoryview]) -> EncryptedBlob:
        """
        Encrypt raw bytes and return an EncryptedBlob.
        """
        if plaintext is None:
            raise ValueError("plaintext must not be None.")
        if not isinstance(plaintext, (bytes, bytearray, memoryview)):
            raise TypeError(f"plaintext must be bytes-like, got {type(plaintext).__name__}.")
        pt = bytes(plaintext)
        if len(pt) == 0:
            raise ValueError("plaintext is empty.")

        try:
            token = self._fernet.encrypt(pt)
            return EncryptedBlob(token=token, created_at=_utc_now_iso())
        except Exception as e:
            raise RuntimeError(f"Encryption failed: {e}") from e

    def decrypt_bytes(self, encrypted: Union[EncryptedBlob, bytes, bytearray, memoryview, str]) -> bytes:
        """
        Decrypt an EncryptedBlob or a Fernet token.

        Accepts:
          - EncryptedBlob
          - token bytes
          - token str (urlsafe base64)
        """
        if encrypted is None:
            raise ValueError("encrypted must not be None.")

        token: Optional[bytes] = None
        if isinstance(encrypted, EncryptedBlob):
            token = encrypted.token
        elif isinstance(encrypted, str):
            token = encrypted.encode("utf-8")
        elif isinstance(encrypted, (bytes, bytearray, memoryview)):
            token = bytes(encrypted)
        else:
            raise TypeError(f"encrypted must be EncryptedBlob/bytes/str, got {type(encrypted).__name__}.")

        if not token:
            raise ValueError("Encrypted token is empty.")

        try:
            return self._fernet.decrypt(token)
        except InvalidToken as e:
            raise ValueError("Decryption failed: invalid token or wrong key.") from e
        except Exception as e:
            raise RuntimeError(f"Decryption failed: {e}") from e

    # ---------
    # Encryption/decryption for JSON/dicts
    # ---------

    def encrypt_json(self, obj: Any) -> EncryptedBlob:
        """
        Encrypt a JSON-serializable object. Stored as UTF-8 JSON.
        """
        s = _safe_json_dumps(obj)
        return self.encrypt_bytes(s.encode("utf-8"))

    def decrypt_json(self, encrypted: Union[EncryptedBlob, bytes, str]) -> Any:
        """
        Decrypt and parse a JSON object.
        """
        pt = self.decrypt_bytes(encrypted)
        return _safe_json_loads(pt)

    def encrypt_dict(self, dct: Mapping[str, Any]) -> EncryptedBlob:
        """
        Encrypt a dict-like object (JSON).
        """
        if dct is None:
            raise ValueError("dct must not be None.")
        if not _is_mapping(dct):
            raise TypeError(f"dct must be a mapping/dict-like object, got {type(dct).__name__}.")
        return self.encrypt_json(dict(dct))

    def decrypt_dict(self, encrypted: Union[EncryptedBlob, bytes, str]) -> Dict[str, Any]:
        """
        Decrypt and return a dict.
        """
        obj = self.decrypt_json(encrypted)
        if not isinstance(obj, dict):
            raise ValueError(f"Decrypted JSON is not an object/dict. Got: {type(obj).__name__}")
        return obj

    # ---------
    # Convenience: redact then encrypt
    # ---------

    def redact_and_encrypt_report(self, report_dict: Mapping[str, Any]) -> EncryptedBlob:
        """
        Redact a report for sharing/storage then encrypt it.
        """
        redacted = redact_report(report_dict)
        return self.encrypt_dict(redacted)

    # ---------
    # Token representation helpers
    # ---------

    @staticmethod
    def token_to_str(encrypted: Union[EncryptedBlob, bytes, bytearray, memoryview]) -> str:
        """
        Convert encrypted token to a storable string.
        """
        if isinstance(encrypted, EncryptedBlob):
            token = encrypted.token
        elif isinstance(encrypted, (bytes, bytearray, memoryview)):
            token = bytes(encrypted)
        else:
            raise TypeError(f"encrypted must be EncryptedBlob or bytes-like, got {type(encrypted).__name__}.")
        if not token:
            raise ValueError("Token is empty.")
        try:
            return token.decode("utf-8")
        except Exception as e:
            raise ValueError(f"Token is not valid UTF-8: {e}") from e

    @staticmethod
    def token_from_str(token_str: str) -> bytes:
        """
        Convert a stored token string back to bytes.
        """
        if not isinstance(token_str, str) or not token_str.strip():
            raise ValueError("token_str must be a non-empty string.")
        return token_str.encode("utf-8")