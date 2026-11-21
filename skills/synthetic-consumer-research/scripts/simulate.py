#!/usr/bin/env python3
"""Ask an LLM to answer a survey question as each synthetic consumer, in free text.

Every persona gets its own fresh LLM call, so answers don't influence each other.
It uses whichever agent CLI is installed: claude (Claude Code), codex (Codex) or gemini (Gemini CLI).
In Cursor, Antigravity or anywhere else, pass --llm-cmd with any command that reads a prompt on stdin.

usage:
  simulate.py --concept concept.md --personas personas.jsonl [--question "..."]
              [--samples 2] [--model sonnet] [--workers 8] [--out responses.jsonl]
  simulate.py ... --llm-cmd 'llm -m gpt-4o'   # any command that reads a prompt on stdin

personas.jsonl: one JSON object per line, e.g.
  {"id": "p01", "age": 34, "age_band": "25-34", "gender": "female", "income": "middle", "region": "Pune, India"}
Every field except "id" is shown to the model as the consumer's profile.
"""
import argparse, json, shlex, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

SYSTEM = "Reply briefly to any questions posed to you."  # the paper's system prompt

PROMPT = """You are taking part in a consumer research survey. Answer as this real person would,
with their budget, habits and priorities. Do not mention that you are an AI.

About you:
{profile}

The product concept you are shown:
---
{concept}
---

Question: {question}

{system} Start with one short sentence that answers the question directly, then add one or two sentences on why. Use your own words. Do not give a number or a rating."""


def default_cmd(model):
    if shutil.which("claude"):
        return ["claude", "-p"] + (["--model", model] if model else [])
    if shutil.which("codex"):
        return ["codex", "exec", "--skip-git-repo-check", "-"] + (["-m", model] if model else [])
    if shutil.which("gemini"):
        return ["gemini"] + (["-m", model] if model else [])
    sys.exit("No agent CLI found (claude, codex or gemini). Pass --llm-cmd, e.g. --llm-cmd 'llm -m gpt-4o'")


def ask(cmd, prompt):
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=300)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[:300])
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", required=True, help="text or markdown file describing the product")
    ap.add_argument("--personas", required=True)
    ap.add_argument("--question", default="How likely are you to purchase the product?")
    ap.add_argument("--samples", type=int, default=2, help="answers per persona (the paper used 2)")
    ap.add_argument("--model", help="model name passed to the detected CLI, e.g. sonnet or haiku for claude")
    ap.add_argument("--llm-cmd", help="any command that reads the prompt on stdin, instead of the detected CLI")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="responses.jsonl")
    a = ap.parse_args()

    cmd = shlex.split(a.llm_cmd) if a.llm_cmd else default_cmd(a.model)
    concept = open(a.concept).read().strip()
    personas = [json.loads(l) for l in open(a.personas) if l.strip()]
    jobs = []
    for i, p in enumerate(personas):
        p.setdefault("id", f"p{i + 1:03d}")
        profile = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in p.items() if k != "id")
        prompt = PROMPT.format(profile=profile, concept=concept, question=a.question, system=SYSTEM)
        jobs += [(p, s, prompt) for s in range(a.samples)]

    def run(job):
        p, s, prompt = job
        try:
            return {**p, "sample": s, "question": a.question, "response": ask(cmd, prompt)}
        except Exception as e:
            print(f"  {p['id']} sample {s} failed: {e}", file=sys.stderr)
            return None

    done = 0
    with open(a.out, "w") as f, ThreadPoolExecutor(a.workers) as pool:
        for row in pool.map(run, jobs):
            if row:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                done += 1
    print(f"{done}/{len(jobs)} answers saved to {a.out}")


if __name__ == "__main__":
    main()
