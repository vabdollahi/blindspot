"""Generate Markdown and figures exclusively from evaluation result JSON."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from eval.schema import (
    EvaluationResult,
    JEPAProbeResult,
    JEPARolloutResult,
    MetricEstimate,
)
from eval.stats import wilson_interval

_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_JEPA_DIR = "jepa"
_ENCODER_NAMES = {
    "random-vit-t16": "Random ViT-T (untrained)",
    "jepa-vit-t16": "JEPA ViT-T (in-domain)",
    "supervised-vit-t16": "Supervised ViT-T",
    "dinov2-vits14": "DINOv2 ViT-S (web-pretrained)",
}
# Headline bars: one linear probe per encoder, plus the supervised model's own output.
_HEADLINE_ROWS = {
    ("random-vit-t16", "global", "linear"): ("Random ViT-T (untrained)", "#9aa5b1"),
    ("jepa-vit-t16", "global", "linear"): ("JEPA ViT-T (in-domain)", "#2f6fb0"),
    ("supervised-vit-t16", "global", "linear"): (
        "Supervised ViT-T, linear probe",
        "#a993c9",
    ),
    ("supervised-vit-t16", "output", "trained"): (
        "Supervised ViT-T, own output (ceiling)",
        "#7b5ea7",
    ),
    ("dinov2-vits14", "global", "linear"): ("DINOv2 ViT-S (web-pretrained)", "#d17a22"),
}
_ROLLOUT_NAMES = {
    "predictor": "JEPA predictor",
    "persistence": "Persistence (nothing moves)",
    "probe_floor": "Probe on the true future (best possible)",
}
_PROBE_COLUMNS = (
    ("cube_xyz_cm", "Cube xyz (cm)"),
    ("cube_xyz_resting_cm", "Cube, resting (cm)"),
    ("cube_xyz_held_cm", "Cube, held (cm)"),
    ("cube_yaw_deg", "Cube yaw (°)"),
    ("bowl_xy_cm", "Bowl xy (cm)"),
    ("tcp_xyz_cm", "TCP xyz (cm)"),
    ("gripper_mm", "Gripper (mm)"),
    ("held_balanced_accuracy", "Held (bal. acc.)"),
    ("in_bowl_balanced_accuracy", "In bowl (bal. acc.)"),
)


def build_report(
    results_dir: str | Path = _DEFAULT_RESULTS_DIR,
    *,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Build a Markdown table and success-rate plot from stored JSON results."""
    results_root = Path(results_dir)
    result_files = sorted(
        path for path in results_root.glob("*/*.json") if path.parent.name != _JEPA_DIR
    )
    jepa_dir = results_root / _JEPA_DIR
    if not result_files and not (jepa_dir / "T3-L1-probes.json").is_file():
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
    markdown_lines.extend(_jepa_sections(jepa_dir, output_root))
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    if rows:
        _write_success_plot(rows, figure_path)
    return markdown_path, figure_path


def _format_metric(metric: MetricEstimate | None, *, percent: bool = False) -> str:
    """Format a point estimate and its interval for a Markdown cell."""
    if metric is None:
        return "–"
    scale = 100.0 if percent else 1.0
    text = f"{scale * metric.value:.1f}{'%' if percent else ''}"
    if metric.ci_low is not None and metric.ci_high is not None:
        text += f" [{scale * metric.ci_low:.1f}, {scale * metric.ci_high:.1f}]"
    return text


def _jepa_sections(jepa_dir: Path, output_root: Path) -> list[str]:
    """Render the Phase 4 probe table and figures from the JEPA result JSON."""
    probe_path = jepa_dir / "T3-L1-probes.json"
    if not probe_path.is_file():
        return []
    probes = JEPAProbeResult.model_validate_json(probe_path.read_text(encoding="utf-8"))
    lines = [
        "# In-domain JEPA (Phase 4)",
        "",
        (
            f"Linear and MLP probes on frozen features, scored on `{probes.test_dataset}` "
            f"({probes.test_episodes} evaluation-seed episodes). Errors are means with "
            f"episode-bootstrap 95% intervals ({probes.resamples:,} resamples); rows with "
            "several seeds pool seeds and episodes."
        ),
        "",
        "| Encoder | Readout | Probe | "
        + " | ".join(label for _, label in _PROBE_COLUMNS)
        + " |",
        "| --- | --- | --- | " + " | ".join("---:" for _ in _PROBE_COLUMNS) + " |",
    ]
    test_rows = [row for row in probes.rows if row.split == "test"]
    for row in test_rows:
        cells = [
            _format_metric(
                row.metrics.get(key), percent=key.endswith("balanced_accuracy")
            )
            for key, _ in _PROBE_COLUMNS
        ]
        lines.append(
            f"| {row.encoder} | {row.readout} | {row.probe} | "
            + " | ".join(cells)
            + " |"
        )
    verdict = {True: "passes", False: "fails", None: "not evaluated"}[probes.h1_pass]
    lines.extend(
        [
            "",
            f"H1 (JEPA global linear cube-position CI entirely below the random encoder's): {verdict}.",
            (
                f"Phase 6 T3-L1 student encoder, chosen on `{probes.validation_dataset}`: "
                f"{probes.phase6_encoder or 'not chosen'}."
            ),
            "",
        ]
    )
    _write_headline_plot(test_rows, output_root / "jepa-headline.png")
    _write_probe_plot(test_rows, output_root / "jepa-probes.png")
    lines.extend(
        [
            "![Probe error by encoder](jepa-headline.png)",
            "",
            "![Probe errors for every readout](jepa-probes.png)",
        ]
    )
    if probes.training_curves:
        _write_training_curves(probes, output_root / "jepa-training.png")
        lines.extend(["", "![JEPA training diagnostics](jepa-training.png)"])
    rollout_path = jepa_dir / "T3-L1-rollout.json"
    if rollout_path.is_file():
        rollout = JEPARolloutResult.model_validate_json(
            rollout_path.read_text(encoding="utf-8")
        )
        _write_rollout_plot(rollout, output_root / "jepa-rollout.png")
        h4 = "passes" if rollout.h4_pass else "fails"
        lines.extend(
            [
                "",
                "## Predictor rollouts",
                "",
                (
                    f"{rollout.windows:,} windows per run, decoded with a linear probe on "
                    "the latent. H4 (predictor below persistence for TCP and held-cube "
                    f"position at 0.4-1.0 s): {h4}."
                ),
                "",
                "![Rollout error against horizon](jepa-rollout.png)",
            ]
        )
    lines.append("")
    return lines


def _write_headline_plot(rows: list, figure_path: Path) -> None:
    """Plot each encoder's linear-probe error, and the supervised model's own output."""
    order = list(_HEADLINE_ROWS)
    chosen = [
        row for row in rows if (row.encoder, row.readout, row.probe) in _HEADLINE_ROWS
    ]
    chosen.sort(key=lambda row: order.index((row.encoder, row.readout, row.probe)))
    targets = (
        ("cube_xyz_cm", "Cube position"),
        ("tcp_xyz_cm", "Gripper (TCP) position"),
        ("bowl_xy_cm", "Bowl position"),
    )
    figure, axes = plt.subplots(
        1, len(targets), figsize=(13.0, 3.6), layout="constrained", sharey=True
    )
    positions = np.arange(len(chosen))
    for axis, (key, title) in zip(axes, targets, strict=True):
        for position, row in zip(positions, chosen, strict=True):
            metric = row.metrics[key]
            low = metric.value - (
                metric.ci_low if metric.ci_low is not None else metric.value
            )
            high = (
                metric.ci_high if metric.ci_high is not None else metric.value
            ) - metric.value
            axis.barh(
                position,
                metric.value,
                color=_HEADLINE_ROWS[(row.encoder, row.readout, row.probe)][1],
            )
            axis.errorbar(
                metric.value,
                position,
                xerr=[[low], [high]],
                fmt="none",
                ecolor="#202c39",
                capsize=3,
            )
            axis.text(
                metric.value + high + 0.15,
                position,
                f"{metric.value:.1f}",
                va="center",
                fontsize=9,
            )
        axis.set_title(title)
        axis.set_xlabel("Mean error (cm), lower is better")
        axis.set_xlim(
            0,
            1.25
            * max(row.metrics[key].ci_high or row.metrics[key].value for row in chosen),
        )
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(
        positions,
        [_HEADLINE_ROWS[(row.encoder, row.readout, row.probe)][0] for row in chosen],
    )
    axes[0].invert_yaxis()
    figure.suptitle("Position error on 200 held-out test episodes (bars: 95% interval)")
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)


def _write_probe_plot(rows: list, figure_path: Path) -> None:
    """Plot linear-probe position errors per encoder and readout."""
    targets = (
        ("cube_xyz_cm", "Cube xyz"),
        ("bowl_xy_cm", "Bowl xy"),
        ("tcp_xyz_cm", "TCP xyz"),
    )
    labels = [
        f"{_ENCODER_NAMES.get(row.encoder, row.encoder)}\n"
        + (
            "own trained output"
            if row.probe == "trained"
            else f"{row.readout} readout, {row.probe}"
        )
        for row in rows
    ]
    positions = np.arange(len(rows))
    width = 0.8 / len(targets)
    figure, axis = plt.subplots(
        figsize=(max(8.0, 1.3 * len(rows)), 5.0), layout="constrained"
    )
    for index, (key, name) in enumerate(targets):
        values, lower, upper = [], [], []
        for row in rows:
            metric = row.metrics.get(key)
            value = metric.value if metric else np.nan
            values.append(value)
            lower.append(
                value - metric.ci_low if metric and metric.ci_low is not None else 0.0
            )
            upper.append(
                metric.ci_high - value if metric and metric.ci_high is not None else 0.0
            )
        axis.bar(positions + index * width, values, width, label=name)
        axis.errorbar(
            positions + index * width,
            values,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#202c39",
            capsize=3,
        )
    axis.set_xticks(positions + width * (len(targets) - 1) / 2, labels, fontsize=8)
    axis.set_ylabel("Probe error (cm)")
    axis.set_title("T3-L1 probe error on evaluation-seed episodes (95% CI)")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)


def _write_training_curves(probes: JEPAProbeResult, figure_path: Path) -> None:
    """Plot validation probe score and collapse diagnostics over training."""
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), layout="constrained")
    for run, points in probes.training_curves.items():
        steps = [point.step for point in points]
        label = Path(run).name
        axes[0].plot(steps, [point.probe_score for point in points], label=label)
        axes[1].plot(
            steps, [point.effective_rank or np.nan for point in points], label=label
        )
        axes[2].plot(
            steps, [point.persistence_ratio or np.nan for point in points], label=label
        )
    for axis, title in zip(
        axes,
        (
            "Validation probe score (lower is better)",
            "Latent effective rank",
            "Prediction error relative to persistence",
        ),
        strict=True,
    ):
        axis.set(title=title, xlabel="Step")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)


def _write_rollout_plot(rollout: JEPARolloutResult, figure_path: Path) -> None:
    """Plot decoded TCP and held-cube error against horizon for each model."""
    panels = (
        ("tcp_xyz", "all", "Gripper (TCP) position"),
        ("cube_xyz", "held", "Cube position, while carried"),
        ("cube_xyz", "all", "Cube position, all moments"),
    )
    figure, axes = plt.subplots(
        1, len(panels), figsize=(14.0, 4.0), layout="constrained"
    )
    for axis, (target, phase, title) in zip(axes, panels, strict=True):
        for curve in rollout.curves:
            if (curve.target, curve.phase) != (target, phase):
                continue
            values = np.array([item.value for item in curve.errors_cm])
            lower = np.array(
                [
                    item.ci_low if item.ci_low is not None else np.nan
                    for item in curve.errors_cm
                ]
            )
            upper = np.array(
                [
                    item.ci_high if item.ci_high is not None else np.nan
                    for item in curve.errors_cm
                ]
            )
            axis.plot(
                rollout.horizons_seconds,
                values,
                marker="o",
                label=_ROLLOUT_NAMES[curve.model],
            )
            axis.fill_between(rollout.horizons_seconds, lower, upper, alpha=0.2)
        axis.set(
            title=title,
            xlabel="Prediction horizon (s)",
            ylabel="Error of the decoded position (cm)",
        )
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)


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
