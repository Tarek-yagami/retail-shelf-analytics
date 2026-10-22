"""Command-line entry point: `shelf <command>`."""

from collections import Counter

import typer

from retail_shelf_analytics.config import REPO_ROOT, load_class_config, load_dataset_config
from retail_shelf_analytics.data.audit import count_boxes, render_audit
from retail_shelf_analytics.data.roboflow_import import import_roboflow_export

REPORTS_DIR = REPO_ROOT / "reports"

app = typer.Typer(no_args_is_help=True)


@app.command("import-roboflow")
def import_roboflow() -> None:
    """Rebuild the annotated dataset at full resolution from the Roboflow export."""
    dataset = load_dataset_config()
    splits = import_roboflow_export(dataset, load_class_config())
    sizes = Counter(splits.values())
    typer.echo(f"Wrote {len(splits)} photos to {dataset.output_dir}: {dict(sorted(sizes.items()))}")


@app.command()
def audit() -> None:
    """Write the box counts of the annotated dataset to reports/data_audit.md."""
    report = render_audit(count_boxes(load_dataset_config().output_dir), load_class_config())
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "data_audit.md").write_text(report, encoding="utf-8")
    typer.echo(report)


if __name__ == "__main__":
    app()
