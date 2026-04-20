"""Train the threat classifier from Storm Events CSVs.

Usage:
    python scripts/train_classifier.py \\
        --data-dir data/raw/storm_events \\
        --output models/threat_classifier.joblib \\
        --metrics models/threat_classifier_metrics.json

If --data-dir doesn't contain real Storm Events CSVs, you can train against
the synthetic dataset by passing --synthetic. This is useful for a quick
end-to-end test before doing the real download.

    python scripts/train_classifier.py --synthetic \\
        --output models/threat_classifier.joblib
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import click
from rich.console import Console

# Allow running this script directly without `pip install -e .`
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardian.risk.classifier import ThreatClassifier  # noqa: E402
from guardian.risk.data.storm_events import (  # noqa: E402
    GULF_COAST_STATES,
    load_storm_events,
)
from guardian.risk.data.synthetic import generate_synthetic_events  # noqa: E402


console = Console()


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/raw/storm_events"),
    show_default=True,
    help="Directory containing Storm Events detail CSVs.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("models/threat_classifier.joblib"),
    show_default=True,
)
@click.option(
    "--metrics",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("models/threat_classifier_metrics.json"),
    show_default=True,
)
@click.option(
    "--synthetic",
    is_flag=True,
    help="Use a generated synthetic dataset instead of real CSVs (for testing).",
)
@click.option("--n-synthetic", type=int, default=20000, show_default=True)
def main(
    data_dir: Path,
    output: Path,
    metrics: Path,
    synthetic: bool,
    n_synthetic: int,
) -> None:
    if synthetic:
        console.rule("[bold yellow]Synthetic-data training run")
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "synth.csv"
            generate_synthetic_events(csv_path, n_events=n_synthetic)
            df = load_storm_events(csv_path)
    else:
        console.rule("[bold cyan]Loading Storm Events from {}".format(data_dir))
        df = load_storm_events(data_dir, states=GULF_COAST_STATES)

    console.print(f"Loaded [bold]{len(df):,}[/bold] events.")
    console.print(f"  Severe events: [bold]{int(df['is_severe'].sum()):,}[/bold]")
    console.print(f"  States:        {sorted(df['STATE'].dropna().unique())}")

    console.rule("[bold]Training")
    clf = ThreatClassifier.fit(df)
    assert clf.metrics is not None

    console.rule("[bold]Held-out test metrics")
    m = clf.metrics
    console.print(f"  Train cells: {m.n_train:,}    Test cells: {m.n_test:,}")
    console.print(f"  Positive rate (test): {m.positive_rate_test:.3f}")
    console.print(f"  Accuracy:         {m.accuracy:.3f}")
    console.print(f"  F1:               {m.f1:.3f}")
    console.print(f"  ROC-AUC:          [bold green]{m.roc_auc:.3f}[/bold green]")
    console.print(f"  Average Precision: {m.avg_precision:.3f}")
    console.print(f"  Confusion matrix: {m.confusion_matrix}")
    console.print(f"  Best params:      {m.best_params}")

    clf.save(output)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(m.to_json() + "\n")
    console.print(f"\n[green]Saved model[/green]   -> {output}")
    console.print(f"[green]Saved metrics[/green] -> {metrics}")


if __name__ == "__main__":
    main()
