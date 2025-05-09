#!/usr/bin/env python
"""Run EMAFusion experiment from YAML config."""
import typer
import yaml 
import json 
import uuid
import pathlib
from emafusion import Cascade
app = typer.Typer()

@app.command()
def main(config: pathlib.Path, prompts: pathlib.Path, out: pathlib.Path = "results.jsonl"):
    cfg     = yaml.safe_load(config.read_text())
    prompts = json.loads(prompts.read_text())
    cascade = Cascade.from_config(cfg)
    with out.open("w") as fh:
        for p in prompts:
            ans = cascade.answer(p["prompt"])
            fh.write(json.dumps({
                "id": uuid.uuid4().hex,
                "prompt": p["prompt"],
                "answer": ans,
            }) + "\n")

if __name__ == "__main__":
    app()
