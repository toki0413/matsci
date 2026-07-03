"""Client-side secret scanner for outbound prompts.

Heavily inspired by Claude Code's secretScanner.ts, which in turn curates
high-confidence rules from gitleaks. We only include patterns with distinctive
prefixes and near-zero false-positive rates. The matched secret text is never
logged or returned; only the rule label/id is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SecretMatch:
    """A secret match — value is intentionally NOT stored."""

    rule_id: str
    label: str


# Anthropic prefix assembled at runtime so the literal sequence is not present
# in static source (mirrors the TS implementation).
_ANTHROPIC_PFX = "-".join(["sk", "ant", "api"])

_SECRET_PATTERNS: list[tuple[str, str, str | None]] = [
    # Cloud providers
    (
        "aws-access-token",
        r"\b((?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16})\b",
        None,
    ),
    ("gcp-api-key", r"\b(AIza[\w-]{35})(?:[\x60'\"\s;]|\\[nr]|$)", None),
    (
        "azure-ad-client-secret",
        r"(?:^|[\\'\"\x60\s>=:(,)])([a-zA-Z0-9_~.]{3}\dQ~[a-zA-Z0-9_~.-]{31,34})(?:$|[\\'\"\x60\s<),])",
        None,
    ),
    ("digitalocean-pat", r"\b(dop_v1_[a-f0-9]{64})(?:[\x60'\"\s;]|\\[nr]|$)", None),
    (
        "digitalocean-access-token",
        r"\b(doo_v1_[a-f0-9]{64})(?:[\x60'\"\s;]|\\[nr]|$)",
        None,
    ),
    # AI APIs
    (
        "anthropic-api-key",
        rf"\b({_ANTHROPIC_PFX}03-[a-zA-Z0-9_\-]{{93}}AA)(?:[\x60'\"\s;]|\\[nr]|$)",
        None,
    ),
    (
        "anthropic-admin-api-key",
        r"\b(sk-ant-admin01-[a-zA-Z0-9_\-]{93}AA)(?:[\x60'\"\s;]|\\[nr]|$)",
        None,
    ),
    (
        "openai-api-key",
        (
            r"\b(sk-(?:proj|svcacct|admin)-(?:[A-Za-z0-9_-]{74}|[A-Za-z0-9_-]{58})T3BlbkFJ"
            r"(?:[A-Za-z0-9_-]{74}|[A-Za-z0-9_-]{58})\b|sk-[a-zA-Z0-9]{20}T3BlbkFJ[a-zA-Z0-9]{20})"
            r"(?:[\x60'\"\s;]|\\[nr]|$)"
        ),
        None,
    ),
    ("huggingface-access-token", r"\b(hf_[a-zA-Z]{34})(?:[\x60'\"\s;]|\\[nr]|$)", None),
    # Version control
    ("github-pat", r"ghp_[0-9a-zA-Z]{36}", None),
    ("github-fine-grained-pat", r"github_pat_\w{82}", None),
    ("github-app-token", r"(?:ghu|ghs)_[0-9a-zA-Z]{36}", None),
    ("github-oauth", r"gho_[0-9a-zA-Z]{36}", None),
    ("github-refresh-token", r"ghr_[0-9a-zA-Z]{36}", None),
    ("gitlab-pat", r"glpat-[\w-]{20}", None),
    ("gitlab-deploy-token", r"gldt-[0-9a-zA-Z_\-]{20}", None),
    # Communication
    ("slack-bot-token", r"xoxb-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*", None),
    ("slack-user-token", r"xox[pe](?:-[0-9]{10,13}){3}-[a-zA-Z0-9-]{28,34}", None),
    ("slack-app-token", r"xapp-\d-[A-Z0-9]+-\d+-[a-z0-9]+", re.IGNORECASE),
    ("twilio-api-key", r"SK[0-9a-fA-F]{32}", None),
    (
        "sendgrid-api-token",
        r"\b(SG\.[a-zA-Z0-9=_\-.]{66})(?:[\x60'\"\s;]|\\[nr]|$)",
        None,
    ),
    # Dev tooling
    ("npm-access-token", r"\b(npm_[a-zA-Z0-9]{36})(?:[\x60'\"\s;]|\\[nr]|$)", None),
    ("pypi-upload-token", r"pypi-AgEIcHlwaS5vcmc[\w-]{50,1000}", None),
    (
        "databricks-api-token",
        r"\b(dapi[a-f0-9]{32}(?:-\d)?)(?:[\x60'\"\s;]|\\[nr]|$)",
        None,
    ),
    (
        "hashicorp-tf-api-token",
        r"[a-zA-Z0-9]{14}\.atlasv1\.[a-zA-Z0-9\-_=]{60,70}",
        None,
    ),
    ("pulumi-api-token", r"\b(pul-[a-f0-9]{40})(?:[\x60'\"\s;]|\\[nr]|$)", None),
    (
        "postman-api-token",
        r"\b(PMAK-[a-fA-F0-9]{24}-[a-fA-F0-9]{34})(?:[\x60'\"\s;]|\\[nr]|$)",
        None,
    ),
    # Observability
    (
        "grafana-api-key",
        r"\b(eyJrIjoi[A-Za-z0-9+/]{70,400}={0,3})(?:[\x60'\"\s;]|\\[nr]|$)",
        None,
    ),
    (
        "grafana-cloud-api-token",
        r"\b(glc_[A-Za-z0-9+/]{32,400}={0,3})(?:[\x60'\"\s;]|\\[nr]|$)",
        None,
    ),
    (
        "grafana-service-account-token",
        r"\b(glsa_[A-Za-z0-9]{32}_[A-Fa-f0-9]{8})(?:[\x60'\"\s;]|\\[nr]|$)",
        None,
    ),
    ("sentry-user-token", r"\b(sntryu_[a-f0-9]{64})(?:[\x60'\"\s;]|\\[nr]|$)", None),
    (
        "sentry-org-token",
        r"\bsntrys_eyJpYXQiO[a-zA-Z0-9+/]{10,200}"
        r"(?:LCJyZWdpb25fdXJs|InJlZ2lvbl91cmwi|cmVnaW9uX3VybCI6)[a-zA-Z0-9+/]{10,200}={0,2}"
        r"_[a-zA-Z0-9+/]{43}",
        None,
    ),
    # Payments
    (
        "stripe-access-token",
        r"\b((?:sk|rk)_(?:test|live|prod)_[a-zA-Z0-9]{10,99})(?:[\x60'\"\s;]|\\[nr]|$)",
        None,
    ),
    ("shopify-access-token", r"shpat_[a-fA-F0-9]{32}", None),
    ("shopify-shared-secret", r"shpss_[a-fA-F0-9]{32}", None),
    # Crypto
    (
        "private-key",
        r"-----BEGIN[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----[\s\S-]{64,}?"
        r"-----END[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----",
        re.IGNORECASE,
    ),
]


_PII_PATTERNS: list[tuple[str, str, str | None]] = [
    # PII — personally identifiable information
    # Email addresses (standard RFC 5322 simplified)
    (
        "pii-email",
        r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b",
        None,
    ),
    # Phone numbers (international format, common patterns)
    (
        "pii-phone",
        r"\b(\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4})\b",
        None,
    ),
    # ORCID researcher ID (0000-0000-0000-000X)
    (
        "pii-orcid",
        r"\b(\d{4}-\d{4}-\d{4}-\d{3}[\dX])\b",
        None,
    ),
    # Chinese ID card (18 digit)
    (
        "pii-cn-idcard",
        r"\b(\d{17}[\dXx])\b",
        None,
    ),
    # Chinese phone number (11 digit mobile)
    (
        "pii-cn-mobile",
        r"\b(1[3-9]\d{9})\b",
        None,
    ),
]


_SPECIAL_CASE_LABELS: dict[str, str] = {
    "aws": "AWS",
    "gcp": "GCP",
    "api": "API",
    "pat": "PAT",
    "ad": "AD",
    "tf": "TF",
    "oauth": "OAuth",
    "npm": "NPM",
    "pypi": "PyPI",
    "jwt": "JWT",
    "github": "GitHub",
    "gitlab": "GitLab",
    "openai": "OpenAI",
    "digitalocean": "DigitalOcean",
    "huggingface": "HuggingFace",
    "hashicorp": "HashiCorp",
    "sendgrid": "SendGrid",
}


def _label_for_rule(rule_id: str) -> str:
    parts = rule_id.replace("-", " ").split()
    return " ".join(_SPECIAL_CASE_LABELS.get(p, p.capitalize()) for p in parts)


class SecretScanner:
    """Scan text for high-confidence secrets and optionally redact them."""

    def __init__(self) -> None:
        self._rules: list[tuple[str, re.Pattern[str]]] = []
        for rule_id, source, flags in _SECRET_PATTERNS:
            self._rules.append((rule_id, re.compile(source, flags or 0)))

        self._pii_patterns: list[tuple[str, re.Pattern[str]]] = []
        for rule_id, source, flags in _PII_PATTERNS:
            self._pii_patterns.append((rule_id, re.compile(source, flags or 0)))

    def scan(self, text: str) -> list[SecretMatch]:
        """Return one match per rule that fired. Secret values are not included."""
        matches: list[SecretMatch] = []
        seen: set[str] = set()
        for rule_id, pattern in self._rules:
            if rule_id in seen:
                continue
            if pattern.search(text):
                seen.add(rule_id)
                matches.append(
                    SecretMatch(rule_id=rule_id, label=_label_for_rule(rule_id))
                )
        return matches

    def redact(self, text: str) -> str:
        """Replace any captured secret with [REDACTED]."""
        for _rule_id, pattern in self._rules:

            def _repl(match: re.Match[str]) -> str:
                full = match.group(0)
                # If the pattern has a capturing group, redact only the group
                # while preserving boundary characters.
                if match.lastindex:
                    g1 = match.group(1)
                    if g1 is not None:
                        return full.replace(g1, "[REDACTED]", 1)
                return "[REDACTED]"

            text = pattern.sub(_repl, text)
        return text

    def scan_pii(self, text: str) -> list[SecretMatch]:
        """Scan only for PII patterns (email, phone, ORCID, etc.)."""
        matches = []
        for rule_id, pattern in self._pii_patterns:
            if pattern.search(text):
                matches.append(SecretMatch(rule_id=rule_id, label=rule_id))
        return matches

    def redact_pii(self, text: str) -> str:
        """Replace PII patterns with [PII_REDACTED]."""
        result = text
        for rule_id, pattern in self._pii_patterns:
            result = pattern.sub("[PII_REDACTED]", result)
        return result


# Module-level singleton for convenience.
_default_scanner: SecretScanner | None = None


def get_scanner() -> SecretScanner:
    """Return the default secret scanner."""
    global _default_scanner
    if _default_scanner is None:
        _default_scanner = SecretScanner()
    return _default_scanner


def scan_for_secrets(text: str) -> list[SecretMatch]:
    """Convenience wrapper around the default scanner."""
    return get_scanner().scan(text)


def redact_secrets(text: str) -> str:
    """Convenience wrapper around the default scanner."""
    return get_scanner().redact(text)
