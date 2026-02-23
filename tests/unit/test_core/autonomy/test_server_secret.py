"""Tests for server_secret.py: HMAC integrity for gate evidence.

Coverage:
- Secret generation, caching, file permissions (0o600)
- Environment variable override (FOUNDRY_MCP_GATE_SECRET)
- Cache invalidation via clear_secret_cache()
- Checksum determinism and different-input divergence
- Versioned payload verification (v1: prefix)
- Legacy checksum backward compatibility
- Delimiter collision resistance
- Empty/invalid checksum rejection
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import os
import stat
from unittest.mock import patch

import pytest

from foundry_mcp.core.autonomy.server_secret import (
    ENV_GATE_SECRET,
    SECRET_FILE_MODE,
    clear_secret_cache,
    compute_integrity_checksum,
    generate_secret,
    get_server_secret,
    load_or_create_secret,
    verify_integrity_checksum,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _clean_secret_cache():
    """Clear the secret cache before and after each test."""
    clear_secret_cache()
    yield
    clear_secret_cache()


@pytest.fixture()
def secret_dir(tmp_path):
    """Provide a temp data dir for secret storage."""
    data_dir = tmp_path / ".foundry-mcp"
    data_dir.mkdir()
    with patch.dict(os.environ, {"FOUNDRY_DATA_DIR": str(data_dir)}):
        yield data_dir


# =============================================================================
# TestGetServerSecret
# =============================================================================


class TestGetServerSecret:
    """Secret generation, caching, file permissions, env override, cache invalidation."""

    def test_generate_secret_returns_32_bytes(self):
        secret = generate_secret()
        assert isinstance(secret, bytes)
        assert len(secret) == 32

    def test_load_or_create_creates_file(self, secret_dir):
        secret = load_or_create_secret()
        assert isinstance(secret, bytes)
        assert len(secret) >= 32

        secret_path = secret_dir / ".server_secret"
        assert secret_path.exists()

    def test_file_permissions_0600(self, secret_dir):
        load_or_create_secret()
        secret_path = secret_dir / ".server_secret"
        mode = stat.S_IMODE(secret_path.stat().st_mode)
        assert mode == SECRET_FILE_MODE, f"Expected 0o600, got {oct(mode)}"

    def test_secret_is_cached(self, secret_dir):
        s1 = get_server_secret()
        s2 = get_server_secret()
        assert s1 == s2

    def test_env_var_override(self, secret_dir):
        test_secret = "my-deterministic-test-secret"
        with patch.dict(os.environ, {ENV_GATE_SECRET: test_secret}):
            secret = get_server_secret()
            assert secret == test_secret.encode()

    def test_cache_invalidation(self, secret_dir):
        s1 = get_server_secret()
        clear_secret_cache()
        # After clearing, should reload (same file, same value)
        s2 = get_server_secret()
        assert s1 == s2

    def test_env_var_takes_precedence_over_file(self, secret_dir):
        """Even if a file-based secret exists, env var wins."""
        # Create a file-based secret first
        load_or_create_secret()
        clear_secret_cache()

        test_secret = "env-override-secret"
        with patch.dict(os.environ, {ENV_GATE_SECRET: test_secret}):
            secret = get_server_secret()
            assert secret == test_secret.encode()

    def test_regenerates_if_file_too_short(self, secret_dir):
        """If existing secret file is < 32 bytes, regenerate."""
        secret_path = secret_dir / ".server_secret"
        secret_path.write_bytes(b"short")
        secret = load_or_create_secret()
        assert len(secret) >= 32


# =============================================================================
# TestComputeIntegrityChecksum
# =============================================================================


class TestComputeIntegrityChecksum:
    """Checksum determinism, input divergence, versioned payload."""

    def test_deterministic(self):
        with patch.dict(os.environ, {ENV_GATE_SECRET: "test-secret"}):
            c1 = compute_integrity_checksum("gate-1", "step-1", "phase-1", "pass")
            c2 = compute_integrity_checksum("gate-1", "step-1", "phase-1", "pass")
            assert c1 == c2

    def test_different_inputs_diverge(self):
        with patch.dict(os.environ, {ENV_GATE_SECRET: "test-secret"}):
            c1 = compute_integrity_checksum("gate-1", "step-1", "phase-1", "pass")
            c2 = compute_integrity_checksum("gate-1", "step-1", "phase-1", "fail")
            assert c1 != c2

    def test_different_gate_ids_diverge(self):
        with patch.dict(os.environ, {ENV_GATE_SECRET: "test-secret"}):
            c1 = compute_integrity_checksum("gate-1", "step-1", "phase-1", "pass")
            c2 = compute_integrity_checksum("gate-2", "step-1", "phase-1", "pass")
            assert c1 != c2

    def test_versioned_payload_format(self):
        """Checksum uses v1: prefix in payload."""
        test_secret = "test-secret"
        with patch.dict(os.environ, {ENV_GATE_SECRET: test_secret}):
            checksum = compute_integrity_checksum("gate-1", "step-1", "phase-1", "pass")

            # Manually compute with v1: prefix to verify
            expected_payload = "v1:gate-1:step-1:phase-1:pass"
            expected = hmac_mod.new(
                test_secret.encode(),
                expected_payload.encode(),
                hashlib.sha256,
            ).hexdigest()
            assert checksum == expected

    def test_returns_hex_string(self):
        with patch.dict(os.environ, {ENV_GATE_SECRET: "test-secret"}):
            checksum = compute_integrity_checksum("g", "s", "p", "v")
            assert isinstance(checksum, str)
            assert len(checksum) == 64  # SHA-256 hex digest length
            assert all(c in "0123456789abcdef" for c in checksum)


# =============================================================================
# TestVerifyIntegrityChecksum
# =============================================================================


class TestVerifyIntegrityChecksum:
    """Verification: valid accepted, invalid rejected, legacy fallback, edge cases."""

    def test_valid_checksum_accepted(self):
        with patch.dict(os.environ, {ENV_GATE_SECRET: "test-secret"}):
            checksum = compute_integrity_checksum("gate-1", "step-1", "phase-1", "pass")
            assert verify_integrity_checksum("gate-1", "step-1", "phase-1", "pass", checksum) is True

    def test_invalid_checksum_rejected(self):
        with patch.dict(os.environ, {ENV_GATE_SECRET: "test-secret"}):
            assert verify_integrity_checksum("gate-1", "step-1", "phase-1", "pass", "bad-checksum") is False

    def test_empty_checksum_rejected(self):
        with patch.dict(os.environ, {ENV_GATE_SECRET: "test-secret"}):
            assert verify_integrity_checksum("gate-1", "step-1", "phase-1", "pass", "") is False

    def test_none_like_checksum_rejected(self):
        """None-ish values should be rejected."""
        with patch.dict(os.environ, {ENV_GATE_SECRET: "test-secret"}):
            # Empty string
            assert verify_integrity_checksum("g", "s", "p", "v", "") is False

    def test_legacy_unversioned_checksum_accepted(self):
        """Legacy checksums (without v1: prefix) should still be accepted during migration."""
        test_secret = "test-secret"
        with patch.dict(os.environ, {ENV_GATE_SECRET: test_secret}):
            # Compute a legacy checksum (no v1: prefix)
            legacy_payload = "gate-1:step-1:phase-1:pass"
            legacy_checksum = hmac_mod.new(
                test_secret.encode(),
                legacy_payload.encode(),
                hashlib.sha256,
            ).hexdigest()

            # verify_integrity_checksum should accept legacy format
            assert verify_integrity_checksum("gate-1", "step-1", "phase-1", "pass", legacy_checksum) is True

    def test_legacy_checksum_rejected_when_env_set(self):
        """Legacy checksums are rejected when FOUNDRY_REJECT_LEGACY_CHECKSUMS=1."""
        test_secret = "test-secret"
        with patch.dict(os.environ, {ENV_GATE_SECRET: test_secret, "FOUNDRY_REJECT_LEGACY_CHECKSUMS": "1"}):
            # Compute a legacy checksum (no v1: prefix)
            legacy_payload = "gate-1:step-1:phase-1:pass"
            legacy_checksum = hmac_mod.new(
                test_secret.encode(),
                legacy_payload.encode(),
                hashlib.sha256,
            ).hexdigest()

            # Should be rejected when env var is set
            assert verify_integrity_checksum("gate-1", "step-1", "phase-1", "pass", legacy_checksum) is False

            # But v1: checksums should still be accepted
            v1_checksum = compute_integrity_checksum("gate-1", "step-1", "phase-1", "pass")
            assert verify_integrity_checksum("gate-1", "step-1", "phase-1", "pass", v1_checksum) is True

    def test_delimiter_collision_known_limitation(self):
        """Colon-delimited payloads can collide when fields contain colons.

        This is a known limitation of the current delimiter-based approach.
        Gate attempt IDs, step IDs, and phase IDs are ULIDs in practice
        and never contain colons, so this is not exploitable.
        """
        with patch.dict(os.environ, {ENV_GATE_SECRET: "test-secret"}):
            # With colons in field values, payloads can collide
            c1 = compute_integrity_checksum("gate:1", "step", "phase", "pass")
            c2 = compute_integrity_checksum("gate", "1:step", "phase", "pass")
            # Same payload string, so checksums match (known limitation)
            assert c1 == c2

            # With ULID-style IDs (no colons), no collision is possible
            c3 = compute_integrity_checksum("01HX1A2B3C", "01HX1A2B3D", "phase-1", "pass")
            c4 = compute_integrity_checksum("01HX1A2B3C", "01HX1A2B3E", "phase-1", "pass")
            assert c3 != c4

    def test_tampered_verdict_rejected(self):
        """A checksum computed with 'pass' should not verify with 'fail'."""
        with patch.dict(os.environ, {ENV_GATE_SECRET: "test-secret"}):
            checksum = compute_integrity_checksum("gate-1", "step-1", "phase-1", "pass")
            assert verify_integrity_checksum("gate-1", "step-1", "phase-1", "fail", checksum) is False

    def test_different_secret_rejects(self):
        """Checksum from one secret should not verify with a different secret."""
        with patch.dict(os.environ, {ENV_GATE_SECRET: "secret-A"}):
            checksum = compute_integrity_checksum("gate-1", "step-1", "phase-1", "pass")

        with patch.dict(os.environ, {ENV_GATE_SECRET: "secret-B"}):
            assert verify_integrity_checksum("gate-1", "step-1", "phase-1", "pass", checksum) is False
