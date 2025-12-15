"""SDD CLI entry point.

JSON-first output for AI coding assistants.

The current CLI emits response-v2 JSON envelopes by default; `--json` is
accepted as an explicit compatibility flag.
"""

import click

from foundry_mcp.cli.config import create_context
from foundry_mcp.cli.registry import register_all_commands


@click.group()
@click.option(
    "--specs-dir",
    envvar="SDD_SPECS_DIR",
    type=click.Path(exists=False),
    help="Override specs directory path",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit JSON response envelopes (default behavior).",
)
@click.pass_context
def cli(ctx: click.Context, specs_dir: str | None, json_output: bool) -> None:
    """SDD CLI - Spec-Driven Development for AI assistants.

    All commands output JSON for reliable parsing by AI coding tools.
    """
    ctx.ensure_object(dict)
    ctx.obj["cli_context"] = create_context(specs_dir=specs_dir)
    ctx.obj["json_output_requested"] = bool(json_output)


# Register all command groups
register_all_commands(cli)


if __name__ == "__main__":
    cli()
