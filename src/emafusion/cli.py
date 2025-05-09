# -----------------------------------------------------------------------------
# src/emafusion/cli.py
# -----------------------------------------------------------------------------
"""Typer‑based CLI entry‑point."""
import pathlib
import typer
from rich import print

from .config import load_config
from .router import TaxonomyRouter, LearnedRouter
from .cascade import Cascade
from .datasets import load_dataset

app = typer.Typer(add_completion=False)


@app.command()
def run(
    config: pathlib.Path = typer.Argument(..., exists=True, help="Path to YAML config"),
    dataset: str = typer.Option(None, help="Built‑in dataset name (jsonl under datasets/)."),
    prompt_file: pathlib.Path = typer.Option(None, help="Path to custom prompts JSONL."),
):
    """Run EMAFusion pipeline on dataset or prompt file."""
    if (dataset is None) == (prompt_file is None):
        typer.echo("Provide exactly one of --dataset or --prompt-file.")
        raise typer.Exit(code=1)

    cfg = load_config(config)

    # Routers --------------------------------------------------------------
    tax_router = TaxonomyRouter(cfg.taxonomy_index)
    lrn_router = LearnedRouter(cfg.learned_ckpt)

    cascade = Cascade.from_config(cfg)

    samples = load_dataset(dataset) if dataset else _load_jsonl(prompt_file)
    for item in samples:
        prompt = item["prompt"] if isinstance(item, dict) else item
        model_name = tax_router.route(prompt)
        # If taxonomy says "ambiguous" fallback to learned
        if model_name == "ambiguous":
            model_name = lrn_router.predict(prompt)
        typer.echo(f"[bold cyan]→ Selecting base model:[/] {model_name}")
        answer = cascade.answer(prompt)
        print(answer)


def _load_jsonl(path: pathlib.Path):
    return [json.loads(l) for l in path.read_text().splitlines()]


if __name__ == "__main__":  # pragma: no cover
    app()
    