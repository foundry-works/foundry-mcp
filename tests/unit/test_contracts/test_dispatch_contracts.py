"""Contract tests for dispatch error envelopes across all unified tool routers.

CONSOLIDATED: The per-router dispatch contract tests have been consolidated
into ``tests/tools/unified/test_dispatch_common.py`` which covers all 16
routers via parametrized fixtures with full-envelope snapshot tests.

This file is kept as a stub to avoid breaking any references.  The shared
assertion helpers can be imported from the canonical location::

    from tests.tools.unified.test_dispatch_common import (
        assert_error_envelope,
        assert_unsupported_action_envelope,
        assert_internal_error_envelope,
    )
"""
