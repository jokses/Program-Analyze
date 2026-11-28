import argparse
import statistics
from pathlib import Path

import checks
import estimates
import extension
import fuzzer
import matplotlib.pyplot as plt
import results
import tables
from evaluation import bound_of
from matplotlib.ticker import FuncFormatter, MaxNLocator

import jpamb
import jvm

SPLITS = ("three", "boundary")
RANDOM = fuzzer.Version.RANDOM
INSTRUCTION = fuzzer.Version.INSTRUCTION
RANGE = fuzzer.Version.RANGE
VERSIONS = (RANGE, INSTRUCTION, RANDOM)
# The width of the text in the ACM acmsmall format, in inches.
TEXT_WIDTH = 5.4

# The inks and colors of the data-viz reference palette, light mode. A figure draws
# the version it is about in blue and the others in gray, each with its own marker,
# so the versions also tell apart in grayscale print.
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
LIGHT_GRAY = "#b5b4ad"
VERSION_STYLE = {
    RANGE: (BLUE, "o"),
    INSTRUCTION: (SECONDARY, "s"),
    RANDOM: (LIGHT_GRAY, "^"),
}
MARKER = {"markersize": 4.5, "markeredgecolor": "white", "markeredgewidth": 0.8}

# The two methods of the coverage curves: one with a range only the extension
# reaches, one with unreachable code.
CURVE_METHODS = (
    "jpamb.cases.Dependent.divisionLoop:(I)V",
    "jpamb.cases.Simple.checkBeforeDivideByN2:(I)I",
)
CURVE_SPLIT = "three"


def use_paper_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Linux Libertine O", "Linux Libertine", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
            "text.color": INK,
            "axes.labelcolor": SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


# No creation date in the PDF, so running the command twice gives the same file.
def save(figure, path: Path):
    metadata = {"CreationDate": None}
    figure.savefig(path, bbox_inches="tight", pad_inches=0.02, metadata=metadata)
    plt.close(figure)


def short(method: str) -> str:
    return method.split(":")[0].rsplit(".", 1)[1]


def version_legend(figure):
    handles = []
    for version in VERSIONS:
        color, marker = VERSION_STYLE[version]
        handles.append(plt.Line2D([], [], color=color, marker=marker, linestyle=""))
    top_center = (0.5, 1.0)
    figure.legend(
        handles, VERSIONS, loc="lower center", ncol=3, bbox_to_anchor=top_center
    )


# Runs on a log axis, written as plain numbers: 1, 10, 100, 1,000.
def plain_log_axis(axis):
    axis.set_xscale("log")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))


# Names on an axis are read, so they take the secondary ink, not the muted one.
def readable_labels(labels):
    for label in labels:
        label.set_color(SECONDARY)


# Figure 1: covered ranges after each run, per version, against the bound. Each
# line is the median over the seeds and the band their lowest and highest. A seed
# that stopped keeps its last value. The marker is where the seeds stop.
def coverage_figure(bc, domain: str, seeds: int, path: Path):
    figure, axes = plt.subplots(1, len(CURVE_METHODS), figsize=(TEXT_WIDTH, 2.0))
    for axis, method in zip(axes, CURVE_METHODS):
        methodid = jvm.AbsMethodID.decode(method)
        bound = frozenset(bound_of(bc, methodid, CURVE_SPLIT, domain))
        for version in reversed(VERSIONS):
            curves = [
                coverage_curve(bc, methodid, version, seed, bound)
                for seed in range(seeds)
            ]
            draw_curves(axis, version, curves)
        axis.axhline(len(bound), color=SECONDARY, linewidth=0.6)
        axis.annotate(
            f"bound, {len(bound)} ranges",
            (1, len(bound)),
            xytext=(2, 3),
            textcoords="offset points",
            color=SECONDARY,
            fontsize=7,
        )
        plain_log_axis(axis)
        axis.set_xlim(1, None)
        axis.set_ylim(0, len(bound) * 1.15)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_title(short(method), loc="left")
        axis.set_xlabel("runs (log scale)")
    axes[0].set_ylabel("covered ranges")
    version_legend(figure)
    save(figure, path)


def coverage_curve(bc, methodid, version, seed: int, bound) -> list[int]:
    covered = []
    fuzzer.fuzz(
        bc,
        methodid,
        version=version,
        seed=seed,
        split=CURVE_SPLIT,
        bound=bound,
        covered_after=covered,
    )
    return covered


def draw_curves(axis, version, curves: list[list[int]]):
    color, marker = VERSION_STYLE[version]
    longest = max(len(curve) for curve in curves)
    padded = [curve + [curve[-1]] * (longest - len(curve)) for curve in curves]
    by_run = list(zip(*padded))
    runs = range(1, longest + 1)
    lowest = [min(values) for values in by_run]
    highest = [max(values) for values in by_run]
    axis.fill_between(runs, lowest, highest, color=color, alpha=0.1, linewidth=0)
    medians = [statistics.median(values) for values in by_run]
    axis.plot(runs, medians, color=color, linewidth=1.2)
    # Where the seeds stop: their median run of stopping, at their median coverage.
    stop = statistics.median(len(curve) for curve in curves)
    final = statistics.median(curve[-1] for curve in curves)
    axis.plot(stop, final, color=color, marker=marker, **MARKER)


# Figure 2: on each method with unreachable code, the runs of every seed of each
# version, and the median as a short bar. The label is how many times the median
# runs of instruction those of range are.
def unreachable_figure(groups_by_split, methods_by_split, path: Path):
    size = (TEXT_WIDTH, 2.4)
    figure, axes = plt.subplots(1, len(SPLITS), figsize=size, sharey=True)
    offsets = {RANGE: -0.22, INSTRUCTION: 0.0, RANDOM: 0.22}
    order = sorted(methods_by_split[SPLITS[0]], key=short)
    for axis, split in zip(axes, SPLITS):
        # The panels share the method axis, so both splits must have the same methods.
        assert sorted(methods_by_split[split], key=short) == order, split
        groups = groups_by_split[split]
        for row, method in enumerate(order):
            for version, offset in offsets.items():
                runs = estimates.runs_of(groups[(method, version)])
                draw_seeds(axis, version, runs, row + offset)
            label_ratio(axis, groups, method, row)
        plain_log_axis(axis)
        axis.set_title(f"split {split}", loc="left")
        axis.set_xlabel("runs until the version stops (log scale)")
        axis.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), [short(method) for method in order])
    readable_labels(axes[0].get_yticklabels())
    axes[0].invert_yaxis()
    version_legend(figure)
    figure.subplots_adjust(wspace=0.35)
    save(figure, path)


def draw_seeds(axis, version, runs: list[int], height: float):
    color, marker = VERSION_STYLE[version]
    heights = [height] * len(runs)
    axis.scatter(runs, heights, color=color, marker=marker, s=9, linewidths=0)
    median = statistics.median(runs)
    axis.plot([median, median], [height - 0.12, height + 0.12], color=color, lw=1.6)


# Right of the axis: how many times the median runs of instruction those of range are.
def label_ratio(axis, groups, method: str, row: int):
    instruction = statistics.median(estimates.runs_of(groups[(method, INSTRUCTION)]))
    range_median = statistics.median(estimates.runs_of(groups[(method, RANGE)]))
    axis.annotate(
        f"{instruction / range_median:.3g}×",
        (1, row),
        xycoords=("axes fraction", "data"),
        xytext=(3, 0),
        textcoords="offset points",
        va="center",
        color=SECONDARY,
        fontsize=7,
    )


# Figure 3: on each method where instruction reaches full instruction coverage, how
# many more ranges a run of range covers than a run of instruction, on average, in
# rank order. The line and band are the mean over the methods and its 95%
# t-interval.
def full_coverage_figure(groups_by_split, methods_by_split, path: Path):
    size = (TEXT_WIDTH, 2.0)
    figure, axes = plt.subplots(1, len(SPLITS), figsize=size, sharey=True)
    for axis, split in zip(axes, SPLITS):
        groups = groups_by_split[split]
        estimate, comparisons = estimates.more_ranges(groups, methods_by_split[split])
        differences = sorted(
            statistics.fmean(comparison.second) - statistics.fmean(comparison.first)
            for comparison in comparisons
        )
        colors = [difference_color(difference) for difference in differences]
        positions = range(1, len(differences) + 1)
        axis.vlines(positions, 0, differences, colors=colors, linewidth=1.0)
        axis.scatter(positions, differences, color=colors, s=7, linewidths=0)
        axis.axhline(0, color=AXIS, linewidth=0.6)
        axis.axhspan(estimate.low, estimate.high, color=BLUE, alpha=0.1, linewidth=0)
        axis.axhline(estimate.mean, color=BLUE, linewidth=0.8)
        label_estimate(axis, estimate)
        axis.set_title(f"split {split}, {len(differences)} methods", loc="left")
        axis.set_xlabel("methods, in rank order")
        axis.set_xticks([])
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("more ranges per run\n(range − instruction)")
    save(figure, path)


# More is blue, fewer is red and the same is gray: the palette's diverging pair.
def difference_color(difference: float) -> str:
    if difference > 0:
        return BLUE
    if difference < 0:
        return RED
    return LIGHT_GRAY


def label_estimate(axis, estimate: estimates.Estimate):
    interval = f"[{estimate.low:.1f}, {estimate.high:.1f}]"
    axis.annotate(
        f"mean {estimate.mean:.1f}, 95% CI {interval}",
        (0.02, 0.95),
        xycoords="axes fraction",
        va="top",
        color=SECONDARY,
        fontsize=7,
    )


# The open ranges the extension tried, by which way reached them. Going backwards and
# the binary search are counted together where both reached a range, so the bar
# shows how far each one adds to the other.
OVERLAP_PARTS = (
    ("only going backwards", BLUE),
    ("both", AQUA),
    ("only the binary search", ORANGE),
    ("only fuzzing for longer", LIGHT_GRAY),
    ("no way", GRID),
)
# A share label fits inside its part from this share of the bar on.
LABEL_FITS = 0.045


def overlap_part(row) -> str:
    backwards = row["backwards"] is not None
    binary = row["binary_search"] is not None
    if backwards and binary:
        return "both"
    if backwards:
        return "only going backwards"
    if binary:
        return "only the binary search"
    if row["longer"] is not None:
        return "only fuzzing for longer"
    return "no way"


# Figure 4: one bar per split, its parts the shares of the open ranges.
def extension_figure(extension_rows_by_split, path: Path):
    figure, axis = plt.subplots(figsize=(TEXT_WIDTH, 1.3))
    for height, split in enumerate(SPLITS):
        draw_overlap_bar(axis, extension_rows_by_split[split], height)
    names = [
        f"{split}\n{len(extension_rows_by_split[split]):,} ranges" for split in SPLITS
    ]
    axis.set_yticks(range(len(SPLITS)), names)
    readable_labels(axis.get_yticklabels())
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    axis.set_xlabel("share of the open ranges the extension tried")
    axis.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in OVERLAP_PARTS]
    figure.legend(
        handles,
        [name for name, _ in OVERLAP_PARTS],
        loc="lower center",
        ncol=len(OVERLAP_PARTS),
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1,
    )
    save(figure, path)


def draw_overlap_bar(axis, targets, height: int):
    counts = {name: 0 for name, _ in OVERLAP_PARTS}
    for row in targets:
        counts[overlap_part(row)] += 1
    start = 0.0
    for name, color in OVERLAP_PARTS:
        share = counts[name] / len(targets)
        axis.barh(
            height,
            share,
            left=start,
            height=0.55,
            color=color,
            edgecolor="white",
            linewidth=1,
        )
        if share > LABEL_FITS:
            # Black on the light parts and white on the dark ones, as the palette says.
            ink = INK if color in (GRID, LIGHT_GRAY, AQUA) else "white"
            middle = start + share / 2
            label = f"{100 * share:.0f}%"
            axis.text(middle, height, label, ha="center", va="center", color=ink)
        start += share


# Everything the figures and tables need from one split of a run.
def load_split(bc, folder: Path, split: str) -> dict:
    rows = checks.read_rows(folder / f"{split}.jsonl")
    _, domain = checks.split_and_domain(rows)
    groups = results.group_runs(rows)
    methods = sorted({method for method, _ in groups})
    extension_rows = checks.read_rows(folder / f"extension_{split}.jsonl")
    return {
        "domain": domain,
        "groups": groups,
        "methods": methods,
        "unreachable": results.unreachable_methods(bc, methods, domain),
        "full": [
            method
            for method in methods
            if results.reaches_full_coverage(groups, method)
        ],
        "extension": extension.target_rows(extension_rows),
        "jvm": tables.jvm_numbers(folder / f"jvm_{split}.txt"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Draw the paper's figures and write its tables from one run."
    )
    parser.add_argument("folder", help="a run's folder, as results/<commit>")
    parser.add_argument("--out", required=True, help="the folder for the files")
    parser.add_argument(
        "--seeds", type=int, default=10, help="seeds per version of the curves"
    )
    args = parser.parse_args()
    folder = Path(args.folder)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})
    use_paper_style()
    run = {split: load_split(bc, folder, split) for split in SPLITS}

    def by_split(key: str) -> dict:
        return {split: run[split][key] for split in SPLITS}

    curve_domain = run[CURVE_SPLIT]["domain"]
    coverage_figure(bc, curve_domain, args.seeds, out / "coverage.pdf")
    groups = by_split("groups")
    unreachable_figure(groups, by_split("unreachable"), out / "unreachable.pdf")
    full_coverage_figure(groups, by_split("full"), out / "full_coverage.pdf")
    extension_figure(by_split("extension"), out / "extension.pdf")
    expected = checks.expected_outcomes()
    tables.claims_table(run, expected, out / "claims.tex")
    tables.own_methods_table(run, out / "own_methods.tex")
