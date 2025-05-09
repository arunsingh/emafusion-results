# -----------------------------------------------------------------------------
# src/emafusion/cli.py
# -----------------------------------------------------------------------------
"""Typer‑based CLI entry‑point for quick experiments."""
from __future__ import annotations

import asyncio
import json
import pathlib
import uuid
from typing import List

import typer  # type: ignore

from .cascade import Cascade
from .config import Settings
from .datasets import load_jsonl

app = typer.Typer(add_help_option=True)


@app.command()
def run(
    config: pathlib.Path = typer.Argument(..., exists=True),
    prompts: pathlib.Path = typer.Argument(..., exists=True),
    out: pathlib.Path = typer.Option("results.jsonl"),
):
    """Run cascade on **prompts** and dump answers → **out** (JSONL)."""

    cfg: Settings = Settings.load(config)
    cascade: Cascade = Cascade.from_config(cfg)
    prompt_list: List[dict] = load_jsonl(prompts)

    async def _process() -> None:  # noqa: D401
        with out.open("w") as fh:
            for item in prompt_list:
                answer = await cascade.answer(item["prompt"])
                fh.write(
                    json.dumps(
                        {
                            "id": uuid.uuid4().hex,
                            "prompt": item["prompt"],
                            "answer": answer,
                        }
                    )
                    + "\n"
                )

    asyncio.run(_process())


if __name__ == "__main__":
    app()
