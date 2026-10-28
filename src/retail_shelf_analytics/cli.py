"""Command-line entry point: `shelf <command>`."""

from collections import Counter

import typer

from retail_shelf_analytics.config import REPO_ROOT, load_class_config, load_dataset_config
from retail_shelf_analytics.data.audit import count_boxes, render_audit
from retail_shelf_analytics.data.roboflow_download import (
    RoboflowClient,
    RoboflowError,
    download_missing_originals,
    download_version,
    load_api_key,
)
from retail_shelf_analytics.data.roboflow_import import import_roboflow_export

REPORTS_DIR = REPO_ROOT / "reports"

app = typer.Typer(no_args_is_help=True)


@app.command("download-roboflow")
def download_roboflow() -> None:
    """Download the configured Roboflow version and the originals of photos missing from data/raw."""
    dataset = load_dataset_config()
    try:
        client = RoboflowClient(load_api_key(), dataset.roboflow)
        download_version(client, dataset.roboflow.version, dataset.roboflow_export_dir)
        saved = download_missing_originals(client, dataset.roboflow_export_dir, dataset.shelf_photos_dir)
    except RoboflowError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Downloaded version {dataset.roboflow.version} and {len(saved)} original photos.")
    typer.echo("Photos whose original is no longer in Roboflow are reported by `shelf import-roboflow`.")


@app.command("import-roboflow")
def import_roboflow() -> None:
    """Rebuild the annotated dataset at full resolution from the Roboflow export."""
    dataset = load_dataset_config()
    splits, without_original = import_roboflow_export(dataset, load_class_config())
    sizes = Counter(splits.values())
    typer.echo(f"Wrote {len(splits)} photos to {dataset.output_dir}: {dict(sorted(sizes.items()))}")
    for name in without_original:
        typer.echo(f"Skipped {name}: no original photo in {dataset.shelf_photos_dir}", err=True)


@app.command()
def audit() -> None:
    """Write the box counts of the annotated dataset to reports/data_audit.md."""
    report = render_audit(count_boxes(load_dataset_config().output_dir), load_class_config())
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "data_audit.md").write_text(report, encoding="utf-8")
    typer.echo(report)


if __name__ == "__main__":
    app()
