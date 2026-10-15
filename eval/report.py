"""Generate Markdown and figures exclusively from evaluation result JSON."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from eval.schema import EvaluationResult
from eval.stats import wilson_interval

_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_report(
    results_dir: str | Path = _DEFAULT_RESULTS_DIR,
    *,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Build a Markdown table and success-rate plot from stored JSON results."""
    results_root = Path(results_dir)
    result_files = sorted(results_root.glob("*/*.json"))
    if not result_files:
        raise ValueError(f"No evaluation result JSON files found under {results_root}")

    evaluations = [
        EvaluationResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in result_files
    ]
    output_root = Path(output_dir) if output_dir is not None else results_root
    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "report.md"
    figure_path = output_root / "success-rates.png"

    rows: list[tuple[EvaluationResult, int, float, tuple[float, float], str]] = []
    for result in evaluations:
        successes = sum(episode.success for episode in result.episodes)
        count = len(result.episodes)
        if count == 0:
            raise ValueError(
                f"Evaluation result {result.method}/{result.task}-{result.level} has no episodes"
            )
        interval = wilson_interval(successes, count)
        failures: dict[str, int] = {}
        for episode in result.episodes:
            if episode.success:
                continue
            reason = episode.failure_reason or "unknown"
            failures[reason] = failures.get(reason, 0) + 1
        failure_text = (
            ", ".join(
                f"{reason}: {failure_count}"
                for reason, failure_count in sorted(failures.items())
            )
            or "none"
        )
        rows.append((result, successes, successes / count, interval, failure_text))

    markdown_lines = [
        "# Evaluation Report",
        "",
        "| Method | Task | Level | Successes | Wilson 95% | Episodes | Failures |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result, successes, rate, interval, failure_text in rows:
        markdown_lines.append(
            f"| {result.method} | {result.task} | {result.level} | "
            f"{successes}/{len(result.episodes)} ({rate:.1%}) | "
            f"{interval[0]:.1%} to {interval[1]:.1%} | "
            f"{len(result.episodes)} | {failure_text} |"
        )
    markdown_lines.extend(
        [
            "",
            "Intervals and failure breakdowns are calculated from the episode records in these JSON files.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    _write_success_plot(rows, figure_path)
    return markdown_path, figure_path


def _write_success_plot(
    rows: list[tuple[EvaluationResult, int, float, tuple[float, float], str]],
    figure_path: Path,
) -> None:
    """Plot success rates with Wilson 95 percent intervals."""
    positions = np.arange(len(rows))
    rates = np.asarray([row[2] for row in rows], dtype=np.float64)
    lower_errors = np.asarray([row[2] - row[3][0] for row in rows], dtype=np.float64)
    upper_errors = np.asarray([row[3][1] - row[2] for row in rows], dtype=np.float64)
    labels = [f"{row[0].method}\n{row[0].task}-{row[0].level}" for row in rows]
    figure_width = max(8.0, 1.2 * len(rows))
    figure, axis = plt.subplots(figsize=(figure_width, 5.0), layout="constrained")
    axis.bar(positions, rates, color="#397f72", width=0.68)
    axis.errorbar(
        positions,
        rates,
        yerr=np.vstack((lower_errors, upper_errors)),
        fmt="none",
        ecolor="#202c39",
        capsize=4,
        linewidth=1.2,
    )
    axis.set_xticks(positions, labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Success rate")
    axis.set_title("Policy evaluation with Wilson 95% intervals")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Parse report paths and generate aggregate artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    markdown_path, figure_path = build_report(
        arguments.results_dir,
        output_dir=arguments.output_dir,
    )
    print(f"Wrote {markdown_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
