from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlsplit

from .base import QuotaAccount, QuotaResult, QuotaWindow, build_unavailable, urlopen_no_redirect

_TIMEOUT_S = 5
_MAX_BYTES = 2 * 1024 * 1024
_MAX_AGE_S = 300
_MAX_ACCOUNTS = 32

FAMILIES = {
    "cli-proxy-claude": "claude",
    "cli-proxy-codex": "codex",
    "cli-proxy-gemini": "gemini",
}


def _config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
    except Exception:  # noqa: BLE001
        return {}
    return config if isinstance(config, dict) else {}


def _usage_url(config: dict[str, Any]) -> Optional[str]:
    section = config.get("proxy_usage")
    url = section.get("url") if isinstance(section, dict) else None
    if not isinstance(url, str):
        return None
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        return None
    if parts.query or parts.fragment or parts.path not in {"", "/"}:
        return None
    return url.strip().rstrip("/") + "/v1/usage"


def _account_names(config: dict[str, Any]) -> dict[str, str]:
    plugins = config.get("plugins")
    entries = plugins.get("entries") if isinstance(plugins, dict) else None
    entry = entries.get("quota") if isinstance(entries, dict) else None
    settings = entry.get("settings") if isinstance(entry, dict) else None
    names = settings.get("accountNames") if isinstance(settings, dict) else None
    if not isinstance(names, dict):
        return {}
    return {
        str(key).strip().lower(): value.strip()[:64]
        for key, value in names.items()
        if len(str(key).strip()) >= 6 and isinstance(value, str) and value.strip()
    }


def _read_usage(url: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    try:
        with urlopen_no_redirect(url, timeout=_TIMEOUT_S) as response:
            raw = response.read(_MAX_BYTES + 1)
    except Exception:  # noqa: BLE001
        return None, "proxy-unreachable"
    if len(raw) > _MAX_BYTES:
        return None, "bad-json"
    try:
        payload = json.loads(raw)
    except ValueError:
        return None, "bad-json"
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        return None, "bad-json"
    try:
        stamp = datetime.fromisoformat(str(payload.get("fetched_at")))
        age = (datetime.now(timezone.utc) - stamp).total_seconds()
    except (TypeError, ValueError):
        return None, "stale-data"
    if not 0 <= age <= _MAX_AGE_S:
        return None, "stale-data"
    return payload, None


def _windows(raw: Any) -> list[QuotaWindow]:
    out = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        used = item.get("used_percent")
        if isinstance(used, bool) or not isinstance(used, (int, float)) or not 0 <= used <= 100:
            continue
        reset = item.get("resets_at")
        out.append(QuotaWindow(
            label=str(item.get("label") or "Quota"),
            used_percent=float(used),
            reset_at=reset if isinstance(reset, str) and reset else None,
        ))
    return out


def _account(entry: dict[str, Any], names: dict[str, str]) -> Optional[QuotaAccount]:
    account_id = entry.get("id")
    if not isinstance(account_id, str) or not account_id.strip():
        return None
    account_id = account_id.strip().lower()
    label = next((name for key, name in names.items() if account_id.startswith(key)), None)
    label = label or account_id[:6]
    if entry.get("state") == "unavailable":
        reason = entry.get("reason")
        return QuotaAccount(id=account_id, label=label, unavailable_reason="proxy-unavailable",
                            details=[reason[:120]] if isinstance(reason, str) and reason else [])
    windows = _windows(entry.get("windows"))
    if not windows:
        return QuotaAccount(id=account_id, label=label, unavailable_reason="no-data")
    return QuotaAccount(id=account_id, label=label, windows=windows)


def fetch_cli_proxy(provider_id: str) -> QuotaResult:
    config = _config()
    url = _usage_url(config)
    if not url:
        return build_unavailable(provider_id, "no-credentials")
    payload, reason = _read_usage(url)
    if payload is None:
        return build_unavailable(provider_id, reason or "no-data")
    providers = payload.get("providers")
    group = providers.get(FAMILIES[provider_id]) if isinstance(providers, dict) else None
    entries = group.get("accounts") if isinstance(group, dict) else None
    if not isinstance(entries, list) or not entries:
        return build_unavailable(provider_id, "no-credentials")
    names = _account_names(config)
    accounts = [a for a in (_account(e, names) for e in entries[:_MAX_ACCOUNTS] if isinstance(e, dict)) if a]
    if not accounts:
        return build_unavailable(provider_id, "no-data")
    result = build_unavailable(provider_id, "no-credentials")
    result.accounts = accounts
    return result


from .registry import register as _register  # noqa: E402

for _provider_id in FAMILIES:
    _register(_provider_id)(lambda _pid=_provider_id: fetch_cli_proxy(_pid))
