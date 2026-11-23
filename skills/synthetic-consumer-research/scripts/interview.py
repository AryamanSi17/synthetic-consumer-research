#!/usr/bin/env python3
"""Turn the strongest objections from a scored run into questions for real customer interviews.

Optional add-on after ssr.py. It reads results/scored.jsonl, takes the most negative answers,
and asks the same agent CLI as simulate.py to write open interview questions that test them.

usage:
  interview.py results [--answers 15] [--questions 6] [--model sonnet] [--llm-cmd '...']
Writes results/interview-questions.md.
"""
import argparse, json, os, shlex
from simulate import ask, default_cmd

PROMPT = """Below are the most negative answers from a simulated consumer survey about a product concept.
They are from AI personas, not real people, so treat them as guesses to check.

Write {n} open questions a founder can ask real customers in an interview to find out whether these
objections are real. Rules: one objection per question, ask about past behaviour and real situations,
never ask "would you buy", no leading questions, plain words. After each question, add in brackets the
objection it tests. Return only the numbered list.

Answers:
{answers}"""


def pick_answers(rows, k):
    rows = sorted(rows, key=lambda r: r["expected_rating"])
    seen, out = set(), []
    for r in rows:  # skip repeated answers from the same persona
        if r.get("id") not in seen:
            seen.add(r.get("id"))
            out.append(r["response"])
        if len(out) == k:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="the folder ssr.py wrote, with scored.jsonl")
    ap.add_argument("--answers", type=int, default=15, help="how many negative answers to read")
    ap.add_argument("--questions", type=int, default=6)
    ap.add_argument("--model")
    ap.add_argument("--llm-cmd")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(a.results, "scored.jsonl")) if l.strip()]
    answers = pick_answers(rows, a.answers)
    cmd = shlex.split(a.llm_cmd) if a.llm_cmd else default_cmd(a.model)
    text = ask(cmd, PROMPT.format(n=a.questions, answers="\n".join(f"- {x}" for x in answers)))
    out = os.path.join(a.results, "interview-questions.md")
    open(out, "w").write("# Questions for real customer interviews\n\n"
                         "Built from the most negative simulated answers. Use them to check the objections with real people.\n\n"
                         + text + "\n")
    print(text + f"\n\nSaved to {out}")


if __name__ == "__main__":
    main()
