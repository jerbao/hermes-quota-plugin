"""Offline tests for the cli-proxy providers and the enabledProviders filter.

cli_proxy reads GET <proxy_usage.url>/v1/usage (schema 1) from a sidecar.
Config and HTTP are mocked; no network. quota_cache._enabled_providers decides
which fetchers refresh_quota_cache runs.

Run from the repo root:  python3 -m unittest tests.test_cli_proxy
"""

from __future__ import annotations

import importlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_stub = types.ModuleType("hermes_constants")
_stub.get_hermes_home = lambda: Path(os.environ.get("HERMES_HOME") or tempfile.gettempdir())
sys.modules.setdefault("hermes_constants", _stub)

_pkg = types.ModuleType("quota_plugin_under_test")
_pkg.__path__ = [str(ROOT)]
sys.modules["quota_plugin_under_test"] = _pkg

cp = importlib.import_module("quota_plugin_under_test.quota_providers.cli_proxy")
qc = importlib.import_module("quota_plugin_under_test.quota_cache")
from quota_plugin_under_test.quota_providers.base import QuotaResult, QuotaWindow  # noqa: E402

URL = "http://proxy.local:8080"
RESET = "2026-11-23T03:10:00+00:00"


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _now(delta_s: float = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=delta_s)).isoformat()


def _payload(**over) -> dict:
    data = {
        "schema": 1,
        "fetched_at": _now(),
        "providers": {
            "claude": {"accounts": [
                {"id": "502e84aaaa", "label": "Claude 502e84", "state": "ready", "resets_at": None,
                 "windows": [{"label": "5 hours", "used_percent": 5.0, "resets_at": RESET, "scope": "account"},
                             {"label": "7 days", "used_percent": 41, "resets_at": None, "scope": "account"}]},
                {"id": "9f1c33bbbb", "state": "ready",
                 "windows": [{"label": "5 hours", "used_percent": 0.0, "resets_at": RESET}]},
            ]},
            "codex": {"accounts": [
                {"id": "7a7a7acccc", "state": "ready",
                 "windows": [{"label": "weekly", "used_percent": 100, "resets_at": RESET}]},
                {"id": "4001a2dddd", "state": "unavailable", "reason": "Credential disabled", "windows": []},
            ]},
            # Removed provider: must be ignored by the plugin.
            "gemini": {"accounts": [
                {"id": "9e91e1eeee", "state": "ready", "windows": [{"label": "d", "used_percent": 1}]},
            ]},
        },
    }
    data.update(over)
    return data


def _config(url=URL, names=None) -> dict:
    cfg: dict = {}
    if url is not None:
        cfg["proxy_usage"] = {"url": url}
    if names is not None:
        cfg["plugins"] = {"entries": {"quota": {"settings": {"accountNames": names}}}}
    return cfg


class _Base(unittest.TestCase):
    def fetch(self, provider="proxy-claude", cfg=None, body=None, raises=None):
        cfg = _config() if cfg is None else cfg
        raw = body if isinstance(body, bytes) else json.dumps(body if body is not None else _payload()).encode()
        opener = mock.Mock(side_effect=raises) if raises else mock.Mock(return_value=_Resp(raw))
        with mock.patch.object(cp, "_config", return_value=cfg), \
                mock.patch.object(cp, "urlopen_no_redirect", opener):
            result = cp.fetch_cli_proxy(provider)
        self.opener = opener
        return result


class ConfigTests(_Base):
    def test_no_url_is_no_credentials_and_makes_no_request(self):
        for cfg in ({}, {"proxy_usage": {}}, {"proxy_usage": {"url": 5}}, _config(url=None)):
            with self.subTest(cfg=cfg):
                r = self.fetch(cfg=cfg)
                self.assertEqual(r.unavailable_reason, "no-credentials")
                self.opener.assert_not_called()

    def test_unsafe_urls_are_rejected_before_any_request(self):
        # path/query/credential/scheme: the sidecar URL must be a bare origin.
        for url in ("http://proxy/api", "http://u:p@proxy:8080", "http://u@proxy", "http://proxy?x=1",
                    "http://proxy#f", "ftp://proxy", "file:///etc/passwd", "proxy:8080", ""):
            with self.subTest(url=url):
                r = self.fetch(cfg=_config(url=url))
                self.assertEqual(r.unavailable_reason, "no-credentials")
                self.opener.assert_not_called()

    def test_trailing_slash_origin_requests_v1_usage(self):
        for url in (URL, URL + "/", "  " + URL + "  "):
            with self.subTest(url=url):
                self.fetch(cfg=_config(url=url))
                self.assertEqual(self.opener.call_args.args[0], URL + "/v1/usage")


class TransportTests(_Base):
    def test_sidecar_down_is_proxy_unreachable(self):
        r = self.fetch(raises=OSError("refused"))
        self.assertEqual(r.unavailable_reason, "proxy-unreachable")
        self.assertFalse(r.accounts)

    def test_bad_json_and_wrong_schema(self):
        cases = {"not json": b"{nope", "array": b"[1]", "schema 2": json.dumps(_payload(schema=2)).encode(),
                 "no schema": json.dumps({"fetched_at": _now(), "providers": {}}).encode(),
                 "schema '1'": json.dumps(_payload(schema="1")).encode(),
                 "oversize": json.dumps(_payload()).encode() + b" " * cp._MAX_BYTES}
        for name, body in cases.items():
            with self.subTest(case=name):
                self.assertEqual(self.fetch(body=body).unavailable_reason, "bad-json")

    def test_stale_future_or_missing_timestamp_is_stale_data(self):
        for name, stamp in {"301s old": _now(-301), "future": _now(+120), "garbage": "yesterday",
                            "none": None, "number": 12345}.items():
            with self.subTest(case=name):
                self.assertEqual(self.fetch(body=_payload(fetched_at=stamp)).unavailable_reason, "stale-data")

    def test_fresh_within_limit_is_accepted(self):
        self.assertEqual(len(self.fetch(body=_payload(fetched_at=_now(-290))).accounts), 2)


class AccountsTests(_Base):
    def test_happy_path_claude_two_accounts_exact_windows(self):
        r = self.fetch("proxy-claude")
        self.assertEqual(r.label, "proxy-claude")
        # The parent row is a placeholder ("no-credentials"); data lives in accounts.
        self.assertEqual(r.unavailable_reason, "no-credentials")
        self.assertEqual([a.id for a in r.accounts], ["502e84aaaa", "9f1c33bbbb"])
        self.assertEqual([a.label for a in r.accounts], ["502e84", "9f1c33"])  # sidecar label ignored
        w = [(x.label, x.used_percent, x.reset_at) for x in r.accounts[0].windows]
        self.assertEqual(w, [("5 hours", 5.0, RESET), ("7 days", 41.0, None)])
        self.assertIsInstance(r.accounts[0].windows[1].used_percent, float)
        self.assertEqual([(x.label, x.used_percent, x.reset_at) for x in r.accounts[1].windows],
                         [("5 hours", 0.0, RESET)])

    def test_each_provider_reads_only_its_own_family(self):
        codex = self.fetch("proxy-codex")
        self.assertEqual([a.id for a in codex.accounts], ["7a7a7acccc", "4001a2dddd"])
        self.assertEqual(codex.accounts[0].windows[0].used_percent, 100.0)

    def test_gemini_is_not_registered_and_its_payload_data_is_ignored(self):
        from quota_plugin_under_test.quota_providers.registry import PROVIDER_FETCHERS as reg
        self.assertEqual(set(cp.FAMILIES), {"proxy-claude", "proxy-codex"})
        self.assertNotIn("proxy-gemini", reg)
        self.assertNotIn("proxy-gemini", cp.FAMILIES)
        for pid in cp.FAMILIES:
            ids = [a.id for a in self.fetch(pid).accounts]
            self.assertNotIn("9e91e1eeee", ids)

    def test_account_names_by_prefix_become_label(self):
        names = {"502": "too short", "502E84": "  Work  ", "9f1c33": "Home", "7a7a7a": "", "4001a2": 7}
        r = self.fetch(cfg=_config(names=names))
        self.assertEqual([a.label for a in r.accounts], ["Work", "Home"])
        self.assertEqual(self.fetch("proxy-codex", cfg=_config(names=names)).accounts[0].label, "7a7a7a")
        self.assertEqual(self.fetch("proxy-codex", cfg=_config(names=names)).accounts[1].label, "4001a2")

    def test_account_name_is_capped_at_64(self):
        r = self.fetch(cfg=_config(names={"502e84": "x" * 100}))
        self.assertEqual(r.accounts[0].label, "x" * 64)

    def test_unavailable_account_carries_reason_in_details(self):
        a = self.fetch("proxy-codex").accounts[1]
        self.assertEqual(a.unavailable_reason, "proxy-unavailable")
        self.assertEqual(a.details, ["Credential disabled"])
        self.assertEqual(a.windows, [])

    def test_unavailable_without_reason_has_no_details_and_long_reason_is_cut(self):
        body = _payload(providers={"codex": {"accounts": [
            {"id": "aaaaaa11", "state": "unavailable"},
            {"id": "bbbbbb22", "state": "unavailable", "reason": "r" * 500}]}})
        a, b = self.fetch("proxy-codex", body=body).accounts
        self.assertEqual(a.details, [])
        self.assertEqual(b.details, ["r" * 120])

    def test_ready_without_valid_windows_is_no_data_and_bad_windows_are_dropped(self):
        body = _payload(providers={"claude": {"accounts": [
            {"id": "aaaaaa11", "state": "ready", "windows": []},
            {"id": "bbbbbb22", "state": "ready", "windows": [
                {"label": "x", "used_percent": True}, {"label": "x", "used_percent": 101},
                {"label": "x", "used_percent": -1}, {"label": "x", "used_percent": "5"}, "junk",
                {"used_percent": 7, "resets_at": ""}]},
            {"id": "", "state": "ready", "windows": [{"used_percent": 1}]},
            "junk", {"state": "ready"}]}})
        accounts = self.fetch(body=body).accounts
        self.assertEqual([a.id for a in accounts], ["aaaaaa11", "bbbbbb22"])
        self.assertEqual(accounts[0].unavailable_reason, "no-data")
        self.assertEqual([(w.label, w.used_percent, w.reset_at) for w in accounts[1].windows], [("Quota", 7.0, None)])

    def test_all_entries_invalid_is_no_data(self):
        body = _payload(providers={"claude": {"accounts": ["junk", {"id": 3}]}})
        self.assertEqual(self.fetch(body=body).unavailable_reason, "no-data")

    def test_missing_family_or_empty_accounts_is_no_credentials(self):
        for name, providers in {"missing": {"codex": {"accounts": [{"id": "aaaaaa"}]}}, "empty": {"claude": {"accounts": []}},
                                "not a dict": {"claude": []}, "providers list": []}.items():
            with self.subTest(case=name):
                r = self.fetch(body=_payload(providers=providers))
                self.assertEqual(r.unavailable_reason, "no-credentials")

    def test_account_count_is_capped(self):
        many = [{"id": "%06x" % i, "windows": [{"used_percent": 1}]} for i in range(cp._MAX_ACCOUNTS + 5)]
        r = self.fetch(body=_payload(providers={"claude": {"accounts": many}}))
        self.assertEqual(len(r.accounts), cp._MAX_ACCOUNTS)

    def test_registered_fetchers_exist_and_return_results(self):
        from quota_plugin_under_test.quota_providers.registry import PROVIDER_FETCHERS as reg
        for pid in cp.FAMILIES:
            self.assertIn(pid, reg)
        with mock.patch.object(cp, "_config", return_value={}):
            self.assertEqual(reg["proxy-codex"]().unavailable_reason, "no-credentials")


class EnabledProvidersTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache = Path(self._tmp.name) / "quota_cache.json"

    def _ok(self, label):
        return lambda: QuotaResult(label=label, windows=[QuotaWindow(label="w", used_percent=1.0)])

    def _sweep(self, fetchers, enabled):
        cfg = {"plugins": {"entries": {"quota": {"settings": {"enabledProviders": enabled}}}}}
        with mock.patch.object(qc, "_cache_path", return_value=str(self.cache)), \
                mock.patch.dict(qc.PROVIDER_FETCHERS, fetchers, clear=True), \
                self._patched_cfg(cfg, fetchers):
            return qc.refresh_quota_cache(budget=2)

    def _patched_cfg(self, cfg, fetchers):
        # hermes_cli is absent in the test env: inject a stub module for the lazy import.
        mod = types.ModuleType("hermes_cli.config")
        mod.load_config_readonly = lambda: cfg
        parent = types.ModuleType("hermes_cli")
        parent.config = mod
        return mock.patch.dict(sys.modules, {"hermes_cli": parent, "hermes_cli.config": mod})

    def test_enabled_providers_parsing(self):
        for value, expected in ((["a", " b ", "", 3], {"a", "b", "3"}), ([], None), (["", "  "], None),
                                ("a", None), (None, None)):
            with self.subTest(value=value):
                cfg = {"plugins": {"entries": {"quota": {"settings": {"enabledProviders": value}}}}}
                with self._patched_cfg(cfg, {}):
                    self.assertEqual(qc._enabled_providers(), expected)

    def test_missing_setting_or_config_error_means_all(self):
        with self._patched_cfg({}, {}):
            self.assertIsNone(qc._enabled_providers())
        mod = types.ModuleType("hermes_cli.config")
        mod.load_config_readonly = mock.Mock(side_effect=RuntimeError("boom"))
        with mock.patch.dict(sys.modules, {"hermes_cli": types.ModuleType("hermes_cli"), "hermes_cli.config": mod}):
            self.assertIsNone(qc._enabled_providers())

    def test_only_listed_fetchers_run(self):
        ran = []
        fetchers = {pid: (lambda p=pid: (ran.append(p), self._ok(p)())[1]) for pid in ("a", "b", "c")}
        cache = self._sweep(fetchers, ["a", "c"])
        self.assertEqual(sorted(ran), ["a", "c"])
        self.assertEqual(sorted(cache["providers"]), ["a", "c"])

    def test_empty_list_runs_everything(self):
        ran = []
        fetchers = {pid: (lambda p=pid: (ran.append(p), self._ok(p)())[1]) for pid in ("a", "b")}
        cache = self._sweep(fetchers, [])
        self.assertEqual(sorted(ran), ["a", "b"])
        self.assertEqual(sorted(cache["providers"]), ["a", "b"])

    def test_unlisted_orphans_are_dropped_but_listed_orphans_are_carried(self):
        self.cache.write_text(json.dumps({"providers": {
            "gone": {"label": "gone", "windows": [{"label": "w", "used_percent": 9}]},
            "gone:acct": {"label": "gone acct", "windows": []},
            "kept": {"label": "kept", "windows": [{"label": "w", "used_percent": 9}]},
            "a": {"label": "a", "windows": []}}}), encoding="utf-8")
        # "kept" is enabled but has no fetcher in this registry -> not-refreshed;
        # "gone" is not enabled -> discarded entirely (also its ":acct" rows).
        cache = self._sweep({"a": self._ok("a"), "b": self._ok("b")}, ["a", "kept"])
        self.assertEqual(sorted(cache["providers"]), ["a", "kept"])
        self.assertEqual(cache["providers"]["kept"]["unavailable_reason"], "not-refreshed")
        self.assertEqual(cache["providers"]["kept"]["windows"], [])
        self.assertIsNone(cache["providers"]["a"]["unavailable_reason"])

    def test_without_setting_orphans_still_carried_as_not_refreshed(self):
        self.cache.write_text(json.dumps({"providers": {
            "gone": {"label": "gone", "windows": [{"label": "w", "used_percent": 9}]}}}), encoding="utf-8")
        cache = self._sweep({"a": self._ok("a")}, None)
        self.assertEqual(cache["providers"]["gone"]["unavailable_reason"], "not-refreshed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
