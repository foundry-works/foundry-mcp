"""Server secret management for gate evidence integrity.

Provides HMAC-based integrity protection for gate evidence using a server secret.
The secret is auto-generated on first use and stored outside the repo tree.

Secret provisioning:
- Auto-generate 32-byte random key on first start if not present
- Store in $FOUNDRY_DATA_DIR/.server_secret (mode 0600)
- Override via FOUNDRY_MCP_GATE_SECRET env var for deterministic testing
- On rotation: in-flight gate evidence becomes invalid - orchestrator
  must re-request gate step (acceptable: gate evidence is short-lived)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import stat
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Environment variable for deterministic testing
ENV_GATE_SECRET = "FOUNDRY_MCP_GATE_SECRET"

# Default data directory (can be overridden)
DEFAULT_DATA_DIR = Path.home() / ".foundry-mcp"

# Secret file name
SECRET_FILENAME = ".server_secret"

# Secret file permissions (owner read/write only)
SECRET_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR  # 0o600


def get_data_dir() -> Path:
    """Get the data directory for storing server secret.

    Resolves in order:
    1. FOUNDRY_DATA_DIR environment variable
    2. Default: ~/.foundry-mcp

    Returns:
        Path to data directory
    """
    env_data_dir = os.environ.get("FOUNDRY_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir)
    return DEFAULT_DATA_DIR


def get_secret_path() -> Path:
    """Get the path to the server secret file.

    Returns:
        Path to server secret file
    """
    return get_data_dir() / SECRET_FILENAME


def _ensure_data_dir() -> None:
    """Ensure the data directory exists."""
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)


def generate_secret() -> bytes:
    """Generate a new 32-byte random secret.

    Returns:
        32-byte random secret
    """
    return secrets.token_bytes(32)


def load_or_create_secret() -> bytes:
    """Load existing secret or create a new one.

    This function:
    1. Checks for FOUNDRY_MCP_GATE_SECRET env var (for testing)
    2. Attempts to load existing secret from file
    3. Creates new secret if none exists

    Returns:
        Server secret as bytes

    Raises:
        OSError: If secret file cannot be created or read
    """
    # Check for environment variable override (for deterministic testing)
    env_secret = os.environ.get(ENV_GATE_SECRET)
    if env_secret:
        logger.debug("Using server secret from environment variable")
        return env_secret.encode()

    secret_path = get_secret_path()

    # Try to load existing secret
    if secret_path.exists():
        try:
            secret = secret_path.read_bytes()
            if len(secret) >= 32:
                logger.debug("Loaded existing server secret from %s", secret_path)
                return secret
            else:
                logger.warning(
                    "Existing secret too short (%d bytes), regenerating",
                    len(secret),
                )
        except OSError as e:
            logger.warning("Failed to read secret file: %s, regenerating", e)

    # Generate new secret
    _ensure_data_dir()
    secret = generate_secret()

    # Write with secure permissions
    try:
        # Create file with secure permissions (write first, then set mode)
        secret_path.write_bytes(secret)
        os.chmod(secret_path, SECRET_FILE_MODE)
        logger.info("Generated new server secret at %s", secret_path)
        return secret
    except OSError as e:
        logger.error("Failed to write secret file: %s", e)
        raise


# Cached secret (lazy-loaded)
_cached_secret: Optional[bytes] = None


def get_server_secret() -> bytes:
    """Get the server secret, loading or creating if necessary.

    The secret is cached after first load for performance.

    Returns:
        Server secret as bytes
    """
    global _cached_secret

    # Always check env var first (in case it changes between calls)
    env_secret = os.environ.get(ENV_GATE_SECRET)
    if env_secret:
        return env_secret.encode()

    if _cached_secret is None:
        _cached_secret = load_or_create_secret()
    return _cached_secret


def clear_secret_cache() -> None:
    """Clear the cached secret.

    Useful for testing or after secret rotation.
    """
    global _cached_secret
    _cached_secret = None


def compute_integrity_checksum(
    gate_attempt_id: str,
    step_id: str,
    phase_id: str,
    verdict: str,
) -> str:
    """Compute HMAC-SHA256 integrity checksum for gate evidence.

    The checksum covers the key fields that define gate evidence identity
    and outcome, preventing tampering with gate results.

    Args:
        gate_attempt_id: Unique gate attempt identifier
        step_id: Step ID this evidence is bound to
        phase_id: Phase ID this gate is for
        verdict: Gate verdict from review

    Returns:
        Hex-encoded HMAC-SHA256 checksum
    """
    secret = get_server_secret()

    # Build payload to sign (deterministic serialization)
    payload = f"{gate_attempt_id}:{step_id}:{phase_id}:{verdict}"

    # Compute HMAC-SHA256
    checksum = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()

    return checksum


def verify_integrity_checksum(
    gate_attempt_id: str,
    step_id: str,
    phase_id: str,
    verdict: str,
    expected_checksum: str,
) -> bool:
    """Verify HMAC-SHA256 integrity checksum for gate evidence.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        gate_attempt_id: Unique gate attempt identifier
        step_id: Step ID this evidence is bound to
        phase_id: Phase ID this gate is for
        verdict: Gate verdict from review
        expected_checksum: Expected hex-encoded checksum

    Returns:
        True if checksum is valid, False otherwise
    """
    if not expected_checksum:
        return False

    computed = compute_integrity_checksum(gate_attempt_id, step_id, phase_id, verdict)

    # Constant-time comparison
    return hmac.compare_digest(computed, expected_checksum)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "get_server_secret",
    "clear_secret_cache",
    "compute_integrity_checksum",
    "verify_integrity_checksum",
    "get_secret_path",
    "get_data_dir",
    "ENV_GATE_SECRET",
]
