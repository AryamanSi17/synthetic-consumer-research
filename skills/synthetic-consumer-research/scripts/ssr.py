#!/usr/bin/env python3
"""Semantic Similarity Rating (SSR): turn free-text survey answers into Likert distributions.

Method from Maier et al. 2025, "LLMs Reproduce Human Purchase Intent via Semantic Similarity
Elicitation of Likert Ratings" (arXiv:2510.08338). Each answer is embedded and compared with
reference sentences for points 1..5. Per reference set:
    p(r) = (s_r - s_min + eps*[r is argmin]) / (sum(s) - 5*s_min + eps)
then the sets are averaged and optionally sharpened with temperature: p(r)^(1/T), renormalised.

usage:
  ssr.py responses.jsonl --anchors anchors/purchase_intent.json [--embedder local|openai]
         [--by age_band] [--human 12,30,80,150,90] [--temperature 1] [--out results]
  ssr.py --selftest
Standard library only, except the embedder: OpenAI needs OPENAI_API_KEY (no package),
local needs `pip install sentence-transformers`.
"""
import argparse, json, math, os, sys, urllib.request
from collections import defaultdict


# ---------- maths ----------
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def set_pmf(sims, eps=0.0):
    """One reference set: similarities for points 1..n -> probabilities (paper Eq. 8)."""
    lo = min(sims)
    num = [s - lo for s in sims]
    num[sims.index(lo)] += eps
    den = sum(sims) - len(sims) * lo + eps
    if den <= 0:  # all anchors equally similar: no information
        return [1 / len(sims)] * len(sims)
    return [x / den for x in num]


def sharpen(p, t):
    if t == 1:
        return p
    q = [x ** (1 / t) for x in p]
    return [x / sum(q) for x in q]


def response_pmf(vec, anchor_sets, eps=0.0, t=1.0):
    pmfs = [set_pmf([cosine(vec, a) for a in s], eps) for s in anchor_sets]
    avg = [sum(col) / len(pmfs) for col in zip(*pmfs)]
    return sharpen(avg, t)


def mean_rating(p):
    return sum((i + 1) * x for i, x in enumerate(p))


def ks_similarity(p, q):
    """1 - max gap between the two cumulative distributions (paper's KS similarity)."""
    cp = cq = gap = 0.0
    for a, b in zip(p, q):
        cp, cq = cp + a, cq + b
        gap = max(gap, abs(cp - cq))
    return 1 - gap


# ---------- embeddings ----------
def embed_openai(texts, model="text-embedding-3-small"):
    key = os.environ.get("OPENAI_API_KEY") or sys.exit("Set OPENAI_API_KEY, or use --embedder local")
    out = []
    for i in range(0, len(texts), 256):
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": model, "input": texts[i:i + 256]}).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out += [d["embedding"] for d in sorted(json.load(r)["data"], key=lambda d: d["index"])]
    return out


def embed_local(texts, model="BAAI/bge-small-en-v1.5"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("Run: pip install sentence-transformers   (or use --embedder openai)")
    return SentenceTransformer(model).encode(texts).tolist()


# ---------- report ----------
def bar(p, width=30):
    return "\n".join(f"  {i + 1}  {'#' * round(x * width):<{width}} {x:6.1%}" for i, x in enumerate(p))


def summarise(rows):
    p = [sum(col) / len(rows) for col in zip(*(r["pmf"] for r in rows))]
    return {"n": len(rows), "pmf": p, "mean": mean_rating(p), "top2": p[3] + p[4]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("responses", nargs="?", help="JSONL with a 'response' field (and persona fields)")
    ap.add_argument("--anchors", default=os.path.join(os.path.dirname(__file__), "..", "anchors", "purchase_intent.json"))
    ap.add_argument("--embedder", choices=["local", "openai"], default="openai" if os.environ.get("OPENAI_API_KEY") else "local")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--by", help="persona field to break results down by, e.g. age_band")
    ap.add_argument("--human", help="real survey counts for ratings 1..5, e.g. 12,30,80,150,90")
    ap.add_argument("--out", default="ssr-results")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.responses:
        ap.error("give a responses.jsonl file")

    anchors = json.load(open(a.anchors))["sets"]
    rows = [json.loads(l) for l in open(a.responses) if l.strip()]
    rows = [r for r in rows if r.get("response", "").strip()]
    if not rows:
        sys.exit("No responses found")
    embed = embed_openai if a.embedder == "openai" else embed_local
    flat = [s for st in anchors for s in st]
    vecs = embed(flat + [r["response"] for r in rows])
    k = len(anchors[0])
    sets = [vecs[i * k:(i + 1) * k] for i in range(len(anchors))]
    for r, v in zip(rows, vecs[len(flat):]):
        r["pmf"] = response_pmf(v, sets, a.epsilon, a.temperature)
        r["expected_rating"] = round(mean_rating(r["pmf"]), 2)

    total = summarise(rows)
    lines = [f"# Synthetic survey results\n", f"Answers scored: {total['n']}  ",
             f"Mean rating: **{total['mean']:.2f} / 5**  ",
             f"Top-2-box (rated 4 or 5): **{total['top2']:.0%}**\n", "```", bar(total["pmf"]), "```"]
    if a.human:
        h = [float(x) for x in a.human.split(",")]
        h = [x / sum(h) for x in h]
        total["ks_similarity_to_human"] = ks_similarity(total["pmf"], h)
        lines += [f"\nReal survey mean: {mean_rating(h):.2f}. KS similarity to the real distribution: "
                  f"**{total['ks_similarity_to_human']:.2f}** (1 = identical; the paper reached 0.88)."]
    if a.by:
        groups = defaultdict(list)
        for r in rows:
            groups[str(r.get(a.by, "unknown"))].append(r)
        total["by"] = {g: summarise(rs) for g, rs in sorted(groups.items())}
        lines += [f"\n## By {a.by}\n", f"| {a.by} | answers | mean | top-2-box |", "|---|---|---|---|"]
        lines += [f"| {g} | {s['n']} | {s['mean']:.2f} | {s['top2']:.0%} |" for g, s in total["by"].items()]
    ranked = sorted(rows, key=lambda r: r["expected_rating"])
    lines += ["\n## Most negative answers\n"] + [f"- ({r['expected_rating']}) {r['response']}" for r in ranked[:5]]
    lines += ["\n## Most positive answers\n"] + [f"- ({r['expected_rating']}) {r['response']}" for r in ranked[-5:][::-1]]

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "scored.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    json.dump(total, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved to {a.out}/ (report.md, summary.json, scored.jsonl)")


def selftest():
    # anchors as orthogonal-ish vectors; a response equal to anchor 5 must lean to 5
    sets = [[[1, 0, 0, 0, 0], [0.7, 0.7, 0, 0, 0], [0, 1, 0.2, 0, 0], [0, 0.3, 1, 0.3, 0], [0, 0, 0.2, 1, 1]]]
    p = response_pmf(sets[0][4], sets)
    assert abs(sum(p) - 1) < 1e-9 and max(p) == p[4] and min(p) == 0, p
    assert abs(sum(sharpen(p, 0.5)) - 1) < 1e-9 and sharpen(p, 0.5)[4] > p[4]
    assert set_pmf([0.5] * 5) == [0.2] * 5
    assert ks_similarity([0.2] * 5, [0.2] * 5) == 1 and abs(ks_similarity([1, 0, 0, 0, 0], [0, 0, 0, 0, 1])) < 1e-9
    assert abs(set_pmf([0.1, 0.2, 0.3, 0.4, 0.5], eps=0.1)[0] - 0.1 / 1.1) < 1e-9
    print("selftest ok")


if __name__ == "__main__":
    main()
