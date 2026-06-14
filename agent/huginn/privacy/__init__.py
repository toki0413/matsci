"""Privacy controls for outbound LLM prompts.

Includes a client-side secret scanner and redaction utilities so that local
files, command output, and memory are not accidentally shipped to cloud LLM
providers verbatim.
"""

from huginn.privacy.scanner import (
    SecretMatch,
    SecretScanner,
    get_scanner,
    redact_secrets,
    scan_for_secrets,
)

__all__ = [
    "SecretMatch",
    "SecretScanner",
    "get_scanner",
    "redact_secrets",
    "scan_for_secrets",
]
