"""``microgem`` command line: score proteins, inspect panels, manage the model.

The CLI is the interface both humans and agents use, so the two things it never
does are: print a probability without its panel's reliability tier, and hide the
fact that precision depends on what fraction of your input is actually that
enzyme class.  ``microgem score --prevalence 0.001`` shows both.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .embed import (
    DEFAULT_MODEL_URL,
    MODEL_NAME,
    cache_dir,
    model_path,
    numpy_weights_dir,
)
from .panels import PANEL_ORDER, PANELS
from .score import (
    default_threshold,
    describe,
    operating_point,
    read_fasta,
    resolve_panels,
    score,
)

TIER_LABEL = {
    "reliable": "reliable",
    "moderate": "moderate",
    "weak": "WEAK",
}
TIER_ORDER = {"reliable": 0, "moderate": 1, "weak": 2}


# ── formatting helpers ─────────────────────────────────────────────────────

def _bar(probability: float, width: int = 20) -> str:
    filled = int(round(probability * width))
    return "#" * filled + "." * (width - filled)


# A score below this is not worth a line in the text report. It is not a
# decision boundary - no panel's cut-off is this low - it is a display floor, so
# that a negative protein prints two rows instead of seventeen rows of zeroes.
DISPLAY_FLOOR = 0.01


def _fmt_panel_table(result, verbose: bool = False) -> str:
    """Show the panels that matter: every call, plus the top few.

    Printing all 17 rows for a protein that scores ~0 everywhere buries the one
    number the reader needs. The rule is: any panel above its cut-off, then the
    highest-scoring remainder down to a display floor, capped at four rows.
    """
    ranked = sorted(result.scores.items(), key=lambda item: -item[1])
    shown: list[tuple[str, float]] = []
    for panel, probability in ranked:
        if panel in result.called or probability >= DISPLAY_FLOOR:
            shown.append((panel, probability))
    if not shown:
        shown = ranked[:1]
    shown = shown[:4] if not verbose else ranked

    lines = []
    for panel, probability in shown:
        info = PANELS[panel]
        state = "CALL" if panel in result.called else "    "
        lines.append(
            f"    {state} {panel:16s} {probability:6.3f} [{_bar(probability)}] "
            f"{TIER_LABEL[info.tier]:8s} AUPRC {info.auprc:.3f}"
        )
    hidden = len(ranked) - len(shown)
    if hidden:
        below = max(probability for _, probability in ranked[len(shown):])
        if below < 0.005:
            lines.append(f"         ... {hidden} further panels, all below 0.005")
        else:
            lines.append(f"         ... {hidden} further panels, all <= {below:.3f} (--verbose for all)")
    return "\n".join(lines)


def _caveat_block(panels: list[str]) -> str:
    notes = []
    for panel in panels:
        info = PANELS[panel]
        if info.caveat:
            notes.append(f"    {panel}: {info.caveat}")
    return "\n".join(notes)


# ── commands ───────────────────────────────────────────────────────────────

def cmd_score(args: argparse.Namespace) -> int:
    panels = resolve_panels(args.panel)
    records = read_fasta(args.fasta)
    if not records:
        print(f"no sequences in {args.fasta}", file=sys.stderr)
        return 1

    if args.limit and len(records) > args.limit:
        print(
            f"{len(records)} sequences, scoring the first {args.limit} (--limit 0 for all).",
            file=sys.stderr,
        )
        records = records[: args.limit]

    result = score(
        dict(records),
        panel=args.panel,
        threshold=args.threshold,
        model=args.model,
        device=args.device,
        backend=args.backend,
        batch_size=args.batch_size,
        progress=not args.quiet,
    )

    if args.format == "json":
        payload = result.to_dict()
        payload["called_panels"] = panels
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.format == "tsv":
        header = ["id", "length", "called", "best_panel", "best_probability", "margin"]
        header += [f"{panel}_prob" for panel in panels]
        print("\t".join(header))
        for item in result.sequences:
            row = [
                item.identifier,
                str(item.length),
                ";".join(item.called) or "-",
                item.best.panel if item.best else "-",
                f"{item.best.probability:.4f}" if item.best else "-",
                f"{item.margin:.4f}" if item.margin is not None else "-",
            ]
            row += [f"{item.scores[panel]:.4f}" for panel in panels]
            print("\t".join(row))
        return 0

    # default: human-readable
    threshold_note = (
        f"one cut-off ({args.threshold:.2f}) applied to every panel"
        if args.threshold is not None
        else "per-panel calibrated cut-offs"
    )
    print(f"microGEM {__version__} | {len(records)} sequences | {len(panels)} panels | {threshold_note}")
    if args.prevalence is not None:
        print(f"assumed prevalence: {args.prevalence:.3%} of input is any one panel's class")
    print()

    for item in result.sequences:
        print(f"> {item.identifier}  ({item.length} aa)")
        print(_fmt_panel_table(item, verbose=args.verbose))
        if item.called:
            best = item.best
            print(f"    -> assigned: {best.panel}  ({PANELS[best.panel].full_name})")
            if len(item.called) > 1:
                print(f"    -> also above cut-off: {', '.join(p for p in item.called if p != best.panel)}")
        elif args.verbose:
            best = item.best
            print(f"    -> no call (best {best.panel} at {best.probability:.3f}, below its cut-off)")
        print()

    if args.show_operating_points:
        _print_operating_points(result, args.prevalence)

    # Only the panels that actually scored matter for the caveats: a panel at
    # 0.001 does not need its limitations explained.
    salient = {item.best.panel for item in result.sequences if item.best}
    for item in result.sequences:
        salient.update(item.called)
    caveats = [panel for panel in PANEL_ORDER if panel in salient and PANELS[panel].caveat]
    if caveats and not args.quiet:
        print("Caveats for the panels that scored:")
        print(_caveat_block(caveats))
        print()

    print(f"called {len(result.hits)}/{len(result.sequences)} sequences")
    return 0


def _print_operating_points(result, prevalence: float | None) -> None:
    """Report precision at the cut-off, at the panel's own prevalence… and yours."""
    print("Operating characteristics at the applied cut-offs")
    print(f"  {'panel':16s} {'cut-off':>7s} {'precision':>9s} {'recall':>7s}", end="")
    if prevalence is not None:
        print(f" {'precision@your prior':>21s}", end="")
    print()
    from .score import _operating_points  # noqa: PLC0415 - presentation-only

    table = _operating_points()
    for panel in result.panels:
        cut = result.cutoffs[panel]
        point = operating_point(panel, cut)
        if point is None:
            continue
        precision, recall = point
        print(f"  {panel:16s} {cut:7.2f} {precision:9.3f} {recall:7.3f}", end="")
        if prevalence is not None:
            row = table[panel]["precision_by_prevalence"]
            nearest_key = min(row, key=lambda key: abs(float(key.rstrip("%")) / 100 - prevalence))
            print(f" {row[nearest_key]:21.3f}", end="")
        print()
    print()


def cmd_panels(args: argparse.Namespace) -> int:
    if args.format == "json":
        print(json.dumps(describe()["panels"], indent=2, ensure_ascii=False))
        return 0
    print(f"microGEM {__version__} — {len(PANELS)} panels\n")
    for name in PANEL_ORDER:
        info = PANELS[name]
        point = operating_point(name, default_threshold(name))
        precision, recall = point if point else (float("nan"), float("nan"))
        print(f"{name}  [{info.family}]  tier: {TIER_LABEL[info.tier]}")
        print(f"  {info.full_name}")
        print(f"  n = {info.n_pos} positives / {info.n_neg} negatives")
        print(f"  cluster-aware OOF: AUC {info.auc:.4f}  AUPRC {info.auprc:.4f}")
        print(f"  cut-off {default_threshold(name):.2f} → precision {precision:.3f}, recall {recall:.3f}")
        if info.caveat:
            print(f"  ! {info.caveat}")
        print()
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    payload = describe()
    payload["model_name"] = MODEL_NAME
    payload["model_paths"] = {
        "torch_checkpoint": _maybe_path(),
        "numpy_weights": str(numpy_weights_dir()),
    }
    payload["cache_dir"] = str(cache_dir())
    if args.format == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    backend = payload["backend"]
    print(f"microGEM {payload['version']}")
    print(f"  embedding backend : {backend}")
    print(f"  model cache       : {payload['cache_dir']}")
    print(f"  ESM-2 checkpoint  : {payload['model_paths']['torch_checkpoint'] or 'not found (run: microgem download)'}")
    print(f"  NumPy weights     : {payload['model_paths']['numpy_weights']}"
          f"{'' if payload['numpy_model'] else ' (not converted)'}")
    print(f"  panels            : {len(payload['panels'])}")
    if backend == "none":
        from .embed import no_backend_help

        print()
        for line in no_backend_help().splitlines():
            print(f"  {line}" if not line.startswith("  ") else line)
    return 0


def _maybe_path():
    try:
        return str(model_path())
    except FileNotFoundError:
        return None


def cmd_download(args: argparse.Namespace) -> int:
    target = Path(args.dest).expanduser() if args.dest else cache_dir() / f"{MODEL_NAME}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    url = args.url or DEFAULT_MODEL_URL

    if target.is_file() and not args.force:
        print(f"already present: {target} ({target.stat().st_size / 2**30:.2f} GiB)")
    else:
        import urllib.request

        print(f"downloading {url}\n  -> {target}", file=sys.stderr)
        tmp = target.with_suffix(".part")
        with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
            total = int(response.headers.get("content-length", 0))
            done = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total and done // (128 << 20) != (done - len(chunk)) // (128 << 20):
                    # Redraw only every 128 MiB. Per-chunk printing turns a
                    # 2.6 GB download into ~2500 lines of progress spam.
                    print(f"\r  {done / 2**30:5.2f} / {total / 2**30:.2f} GiB", end="", file=sys.stderr, flush=True)
        print("", file=sys.stderr)
        tmp.replace(target)
        print(f"saved {target}")

    if args.convert:
        from .numpy_model import convert_checkpoint

        out_dir = numpy_weights_dir()
        print(f"converting to NumPy weights in {out_dir} …", file=sys.stderr)
        convert_checkpoint(target, out_dir)
        print(f"done — torch is no longer needed to score proteins ({out_dir})")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    from .numpy_model import convert_checkpoint

    checkpoint = model_path(args.model)
    out_dir = Path(args.out).expanduser() if args.out else numpy_weights_dir()
    print(f"converting {checkpoint}\n        -> {out_dir}", file=sys.stderr)
    convert_checkpoint(checkpoint, out_dir)
    print("done. verify with: microgem selftest --backend numpy")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Answer one question: does this install reproduce the published scores?

    Positive control: one SwissProt protein per panel that the panel was trained
    on must score above its cut-off.  Negative control: a set of proteins from
    other panels' positives must *not* light up every panel.  Both are cheap
    because they use ``assets/reference.faa``.
    """
    from .selftest import run_selftest

    return run_selftest(args)


# ── argument parsing ───────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microgem",
        description="Score microbial proteins against microGEM's 17 enzyme panels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  microgem score proteins.faa\n"
            "  microgem score proteins.faa --panel GLY --format json\n"
            "  microgem score proteins.faa --panel AATE --threshold 0.5 --show-operating-points\n"
            "  microgem panels\n"
            "  microgem download            # 2.6 GB, once\n"
            "  microgem selftest\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"microGEM {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    score_parser = sub.add_parser("score", help="score a FASTA file")
    score_parser.add_argument("fasta", type=Path, help="protein FASTA (any number of records)")
    score_parser.add_argument("--panel", default=None,
                              help="one panel, a family (FATE-L, AATE, SAM) or 'all' (default)")
    score_parser.add_argument("--threshold", type=float, default=None,
                              help="override every panel's calibrated cut-off (default: per-panel)")
    score_parser.add_argument("--format", choices=("text", "tsv", "json"), default="text")
    score_parser.add_argument("--model", default=None, help="path to esm2_t33_650M_UR50D.pt")
    score_parser.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
    score_parser.add_argument("--backend", choices=("auto", "torch", "numpy"), default="auto")
    score_parser.add_argument("--batch-size", type=int, default=16)
    score_parser.add_argument("--limit", type=int, default=0,
                              help="score only the first N records (0 = all)")
    score_parser.add_argument("--prevalence", type=float, default=None,
                              help="expected fraction of your input that is this enzyme class, "
                                   "e.g. 0.001; shows prevalence-adjusted precision")
    score_parser.add_argument("--show-operating-points", action="store_true",
                              help="print precision/recall at the applied cut-offs")
    score_parser.add_argument("--verbose", action="store_true", help="explain sequences with no call")
    score_parser.add_argument("--quiet", action="store_true", help="suppress progress and caveats")
    score_parser.set_defaults(func=cmd_score)

    panels_parser = sub.add_parser("panels", help="list the 17 panels with their performance")
    panels_parser.add_argument("--format", choices=("text", "json"), default="text")
    panels_parser.set_defaults(func=cmd_panels)

    info_parser = sub.add_parser("info", help="show backend, model and cache locations")
    info_parser.add_argument("--format", choices=("text", "json"), default="text")
    info_parser.set_defaults(func=cmd_info)

    download_parser = sub.add_parser("download", help="fetch the ESM-2 checkpoint")
    download_parser.add_argument("--dest", default=None, help="destination path")
    download_parser.add_argument("--url", default=None, help="mirror URL")
    download_parser.add_argument("--force", action="store_true", help="re-download if present")
    download_parser.add_argument("--convert", action="store_true",
                                 help="also write the NumPy weights, so torch is not needed to score")
    download_parser.set_defaults(func=cmd_download)

    convert_parser = sub.add_parser("convert", help="convert the checkpoint for the NumPy backend")
    convert_parser.add_argument("--model", default=None)
    convert_parser.add_argument("--out", default=None)
    convert_parser.set_defaults(func=cmd_convert)

    selftest_parser = sub.add_parser("selftest", help="verify this install reproduces known scores")
    selftest_parser.add_argument("--backend", choices=("auto", "torch", "numpy"), default="auto")
    selftest_parser.add_argument("--limit", type=int, default=0, help="test the first N reference proteins")
    selftest_parser.set_defaults(func=cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
