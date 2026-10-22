"""Count annotated boxes per split, brand group and label, and render them as a Markdown report."""

from collections import Counter
from pathlib import Path

from retail_shelf_analytics.config import ClassConfig
from retail_shelf_analytics.data.roboflow_import import SPLITS


def count_boxes(dataset_dir: Path) -> dict[str, Counter]:
    """Return, per split, the number of boxes of each label id, plus the photo count under key -1."""
    counts = {}
    for split in SPLITS:
        counter = Counter()
        for label_file in (dataset_dir / "labels" / split).glob("*.txt"):
            counter[-1] += 1
            counter.update(int(line.split()[0]) for line in label_file.read_text().splitlines() if line.strip())
        counts[split] = counter
    return counts


def _row(cells: list[object]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def render_audit(counts: dict[str, Counter], classes: ClassConfig) -> str:
    header = _row(["", *SPLITS])
    divider = _row(["---"] * (len(SPLITS) + 1))
    boxes = {split: sum(n for label_id, n in counts[split].items() if label_id >= 0) for split in SPLITS}
    target_ids = {i for i, label in enumerate(classes.labels) if label.brand == classes.target_brand}

    def share(split: str, ids: set[int]) -> str:
        n = sum(counts[split][i] for i in ids)
        return f"{n} ({100 * n / boxes[split]:.0f}%)" if boxes[split] else "0"

    other_ids = set(range(len(classes.labels))) - target_ids
    lines = [
        "# Annotated data audit",
        "",
        header,
        divider,
        _row(["photos", *(counts[s][-1] for s in SPLITS)]),
        _row(["boxes", *(boxes[s] for s in SPLITS)]),
        _row([f"{classes.target_brand} boxes", *(share(s, target_ids) for s in SPLITS)]),
        _row(["other brands", *(share(s, other_ids) for s in SPLITS)]),
        "",
        "## Boxes per label",
        "",
        header,
        divider,
    ]
    by_total = sorted(range(len(classes.labels)), key=lambda i: -sum(counts[s][i] for s in SPLITS))
    lines += [_row([classes.labels[i].name, *(counts[s][i] for s in SPLITS)]) for i in by_total]
    return "\n".join(lines) + "\n"
