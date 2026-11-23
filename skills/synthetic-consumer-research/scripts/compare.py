#!/usr/bin/env python3
"""Compare scored survey runs side by side, and check whether wording changes the winner.

Optional add-on: run simulate.py and ssr.py as usual for each concept or version, then compare
their result folders here. Nothing about the normal steps changes.

usage:
  compare.py A=results-a B=results-b                      # rank concepts
  compare.py A/plain=ra-plain A/copy=ra-copy \\
             B/plain=rb-plain B/copy=rb-copy                # wording check
  compare.py --selftest

Labels with a slash are read as idea/version. For each version, the ideas are ranked, and the
script warns if a different idea wins under different wording. It also shows how much each
version moves each idea's score. Use the same personas and question in every run.
"""
import json, os, sys
from collections import defaultdict

GAP = 0.2  # the skill's rule: smaller mean differences are not reliable


def load(folder):
    summary = json.load(open(os.path.join(folder, "summary.json")))
    ids = set()
    scored = os.path.join(folder, "scored.jsonl")
    if os.path.exists(scored):
        ids = {json.loads(l).get("id") for l in open(scored) if l.strip()}
    return summary, ids


def report(runs):
    """runs: {label: summary dict}. Returns the report lines."""
    ranked = sorted(runs.items(), key=lambda kv: -kv[1]["mean"])
    lines = ["# Comparison\n", "| run | answers | mean | top-2-box |", "|---|---|---|---|"]
    lines += [f"| {k} | {s['n']} | {s['mean']:.2f} | {s['top2']:.0%} |" for k, s in ranked]
    if len(ranked) > 1:
        gap = ranked[0][1]["mean"] - ranked[1][1]["mean"]
        verdict = "a real lead" if gap > GAP else f"too close to call (under {GAP})"
        lines.append(f"\n**{ranked[0][0]}** leads **{ranked[1][0]}** by {gap:.2f}: {verdict}.")

    grid = defaultdict(dict)  # idea -> version -> mean
    for k, s in runs.items():
        if "/" in k:
            idea, version = k.split("/", 1)
            grid[idea][version] = s["mean"]
    versions = sorted({v for vs in grid.values() for v in vs})
    if grid and len(versions) > 1:
        lines += ["\n## Wording check\n", "| idea | " + " | ".join(versions) + " | swing |",
                  "|---|" + "---|" * (len(versions) + 1)]
        for idea, vs in sorted(grid.items()):
            cells = [f"{vs[v]:.2f}" if v in vs else "-" for v in versions]
            swing = max(vs.values()) - min(vs.values())
            lines.append(f"| {idea} | " + " | ".join(cells) + f" | {swing:.2f} |")
        winners = {}
        for v in versions:
            scored = {i: vs[v] for i, vs in grid.items() if v in vs}
            if len(scored) > 1:
                winners[v] = max(scored, key=scored.get)
        if len(set(winners.values())) > 1:
            lines.append("\n**Warning: the winner changes with the wording** ("
                         + ", ".join(f"{v}: {w}" for v, w in winners.items())
                         + "). The copy may be doing the work, not the idea. Don't pick a build from this score alone.")
        elif winners:
            lines.append(f"\nThe winner stays the same under every wording ({next(iter(winners.values()))}).")
        big = [i for i, vs in grid.items() if max(vs.values()) - min(vs.values()) > GAP]
        if big:
            lines.append(f"Wording alone moves the score by more than {GAP} for: {', '.join(sorted(big))}.")
    return lines


def main():
    args = sys.argv[1:]
    if args == ["--selftest"]:
        return selftest()
    if len(args) < 2 or any("=" not in a for a in args):
        sys.exit(__doc__)
    runs, idsets = {}, {}
    for a in args:
        label, folder = a.split("=", 1)
        runs[label], idsets[label] = load(folder)
    if len({frozenset(s) for s in idsets.values() if s}) > 1:
        print("Note: the runs used different personas, so the comparison is less fair.\n")
    print("\n".join(report(runs)))


def selftest():
    s = lambda m: {"n": 200, "mean": m, "top2": 0.3}
    out = "\n".join(report({"A/plain": s(2.6), "B/plain": s(2.4), "A/copy": s(2.7), "B/copy": s(3.1)}))
    assert "winner changes" in out and "B" in out.split("more than")[-1], out
    out = "\n".join(report({"A/plain": s(3.0), "B/plain": s(2.4), "A/copy": s(3.1), "B/copy": s(2.5)}))
    assert "stays the same" in out and "winner changes" not in out, out
    out = "\n".join(report({"A": s(3.0), "B": s(2.9)}))
    assert "too close" in out and "Wording" not in out, out
    print("selftest ok")


if __name__ == "__main__":
    main()
