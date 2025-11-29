"""SDD CLI entry point.

JSON-only output for AI coding assistants.
"""

import click

from foundry_mcp.cli.config import CLIContext, create_context
from foundry_mcp.cli.registry import register_all_commands


@click.group()
@click.option(
    "--specs-dir",
    envvar="SDD_SPECS_DIR",
    type=click.Path(exists=False),
    help="Override specs directory path",
)
@click.pass_context
def cli(ctx: click.Context, specs_dir: str | None) -> None:
    """SDD CLI - Spec-Driven Development for AI assistants.

    All commands output JSON for reliable parsing by AI coding tools.
    """
    ctx.ensure_object(dict)
    ctx.obj["cli_context"] = create_context(specs_dir=specs_dir)


# Register all command groups
register_all_commands(cli)


if __name__ == "__main__":
    cli()
