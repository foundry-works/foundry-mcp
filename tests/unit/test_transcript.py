"""Tests for transcript parsing module."""

import json
import tempfile
from pathlib import Path


from foundry_mcp.cli.transcript import (
    TokenMetrics,
    find_transcript_by_marker,
    is_clear_command,
    parse_transcript,
)


class TestTokenMetrics:
    """Tests for TokenMetrics dataclass."""

    def test_context_percentage_normal(self):
        """Test context percentage calculation."""
        metrics = TokenMetrics(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=500,
            total_tokens=2000,
            context_length=77500,
        )
        # 77500 / 155000 = 50%
        assert metrics.context_percentage(155000) == 50.0

    def test_context_percentage_zero_max(self):
        """Test context percentage with zero max context."""
        metrics = TokenMetrics(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=500,
            total_tokens=2000,
            context_length=77500,
        )
        assert metrics.context_percentage(0) == 0.0


class TestIsClearCommand:
    """Tests for is_clear_command function."""

    def test_clear_command_string_content(self):
        """Test detecting /clear in string content."""
        entry = {
            "type": "user",
            "message": {
                "content": "<command-name>/clear</command-name>\n<command-message>clear</command-message>"
            },
        }
        assert is_clear_command(entry) is True

    def test_clear_command_list_content(self):
        """Test detecting /clear in list content."""
        entry = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "text", "text": "<command-name>/clear</command-name>"}
                ]
            },
        }
        assert is_clear_command(entry) is True

    def test_non_clear_command(self):
        """Test non-clear commands are not detected."""
        entry = {
            "type": "user",
            "message": {"content": "Hello, how are you?"},
        }
        assert is_clear_command(entry) is False

    def test_assistant_message_not_clear(self):
        """Test assistant messages are not detected as clear."""
        entry = {
            "type": "assistant",
            "message": {"content": "/clear"},
        }
        assert is_clear_command(entry) is False


class TestParseTranscript:
    """Tests for parse_transcript function."""

    def test_parse_empty_file(self):
        """Test parsing an empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            f.flush()

            metrics = parse_transcript(f.name)
            assert metrics is not None
            assert metrics.input_tokens == 0
            assert metrics.context_length == 0

    def test_parse_single_entry(self):
        """Test parsing a single entry with usage data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            entry = {
                "type": "assistant",
                "message": {
                    "usage": {
                        "input_tokens": 1000,
                        "output_tokens": 200,
                        "cache_read_input_tokens": 500,
                        "cache_creation_input_tokens": 100,
                    }
                },
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()

            metrics = parse_transcript(f.name)
            assert metrics is not None
            assert metrics.input_tokens == 1000
            assert metrics.output_tokens == 200
            assert metrics.cached_tokens == 600  # 500 + 100
            assert metrics.context_length == 1600  # 1000 + 500 + 100

    def test_parse_clear_resets_counters(self):
        """Test that /clear command resets token counters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # First entry with usage
            entry1 = {
                "type": "assistant",
                "message": {
                    "usage": {
                        "input_tokens": 10000,
                        "output_tokens": 500,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    }
                },
            }
            # Clear command
            clear_entry = {
                "type": "user",
                "message": {"content": "<command-name>/clear</command-name>"},
            }
            # Second entry after clear
            entry2 = {
                "type": "assistant",
                "message": {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    }
                },
            }
            f.write(json.dumps(entry1) + "\n")
            f.write(json.dumps(clear_entry) + "\n")
            f.write(json.dumps(entry2) + "\n")
            f.flush()

            metrics = parse_transcript(f.name)
            assert metrics is not None
            # Should only have counts from after the clear
            assert metrics.input_tokens == 100
            assert metrics.output_tokens == 50
            assert metrics.context_length == 100

    def test_parse_skips_sidechain(self):
        """Test that sidechain entries are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            sidechain = {
                "type": "assistant",
                "isSidechain": True,
                "message": {
                    "usage": {
                        "input_tokens": 50000,
                        "output_tokens": 1000,
                    }
                },
            }
            regular = {
                "type": "assistant",
                "message": {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                    }
                },
            }
            f.write(json.dumps(sidechain) + "\n")
            f.write(json.dumps(regular) + "\n")
            f.flush()

            metrics = parse_transcript(f.name)
            assert metrics is not None
            # Should only count the regular entry
            assert metrics.input_tokens == 100

    def test_parse_nonexistent_file(self):
        """Test parsing a non-existent file returns None."""
        metrics = parse_transcript("/nonexistent/path/file.jsonl")
        assert metrics is None


class TestFindTranscriptByMarker:
    """Tests for find_transcript_by_marker function."""

    def test_find_marker_in_transcript(self, tmp_path, monkeypatch):
        """Test finding a marker in a transcript file."""
        # Create a test project directory
        project_path = tmp_path / "myproject"
        project_path.mkdir()

        # Encode the path the same way the function does
        project_dir_name = str(project_path).replace("/", "-").replace("_", "-")

        # Create the transcript directory
        transcript_dir = tmp_path / ".claude" / "projects" / project_dir_name
        transcript_dir.mkdir(parents=True)

        # Create a transcript file with a marker
        transcript = transcript_dir / "session.jsonl"
        entry = {
            "type": "user",
            "message": {"content": "SESSION_MARKER_ABC12345"},
        }
        transcript.write_text(json.dumps(entry) + "\n")

        # Mock Path.home() to return tmp_path
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Search from the project path
        result = find_transcript_by_marker(
            project_path,
            "SESSION_MARKER_ABC12345",
            max_retries=1,
            allow_home_search=True,
        )
        assert result is not None
        assert result == transcript

    def test_marker_not_found(self, tmp_path, monkeypatch):
        """Test that None is returned when marker not found."""
        # Create a test project directory
        project_path = tmp_path / "myproject"
        project_path.mkdir()

        # Encode the path the same way the function does
        project_dir_name = str(project_path).replace("/", "-").replace("_", "-")

        # Create the transcript directory
        transcript_dir = tmp_path / ".claude" / "projects" / project_dir_name
        transcript_dir.mkdir(parents=True)

        transcript = transcript_dir / "session.jsonl"
        entry = {"type": "user", "message": {"content": "some other content"}}
        transcript.write_text(json.dumps(entry) + "\n")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = find_transcript_by_marker(
            project_path,
            "SESSION_MARKER_NOTFOUND",
            max_retries=1,
            allow_home_search=True,
        )
        assert result is None

    def test_no_transcript_directory(self, tmp_path, monkeypatch):
        """Test when no transcript directory exists."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = find_transcript_by_marker(
            Path("/nonexistent/path"),
            "SESSION_MARKER_ABC",
            max_retries=1,
            allow_home_search=True,
        )
        assert result is None

    def test_explicit_search_directory(self, tmp_path):
        """Explicit search directories are honored without home scanning."""
        transcripts_dir = tmp_path / "transcripts"
        transcripts_dir.mkdir()
        transcript = transcripts_dir / "session.jsonl"
        transcript.write_text(
            json.dumps({"type": "user", "message": {"content": "SESSION_MARKER_Z"}})
            + "\n"
        )

        result = find_transcript_by_marker(
            Path("/anywhere"),
            "SESSION_MARKER_Z",
            max_retries=1,
            search_dirs=[transcripts_dir],
        )
        assert result == transcript
