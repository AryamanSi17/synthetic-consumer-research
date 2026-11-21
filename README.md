# Synthetic Consumer Research: an Agent Skill for LLM Surveys with Semantic Similarity Rating (SSR)

**Test a product idea on 100 simulated customers in a few minutes.** This is an [Agent Skill](https://docs.claude.com/en/docs/claude-code/skills) for Claude Code (also usable from any terminal) that runs a synthetic consumer survey. It gives LLM personas a product concept, collects their answers in their own words, and turns those answers into realistic 1 to 5 purchase intent ratings with **Semantic Similarity Rating (SSR)**.

SSR comes from the 2025 paper [*LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings*](https://arxiv.org/abs/2510.08338) (Maier, Aslak, Fiaschi et al., PyMC Labs and Colgate-Palmolive). It was tested on 57 real product surveys with 9,300 people. SSR reached **90% of human test-retest reliability**: the consistency real people have when they answer the same survey twice. Its rating spread matched the real one with a **KS similarity of 0.88**.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![Claude Code skill](https://img.shields.io/badge/Claude%20Code-skill-orange)
[![arXiv 2510.08338](https://img.shields.io/badge/arXiv-2510.08338-b31b1b.svg)](https://arxiv.org/abs/2510.08338)

---

## Contents

- [What it does](#what-it-does)
- [Why not just ask the AI for a rating?](#why-not-just-ask-the-ai-for-a-rating)
- [Install](#install)
- [Quick start](#quick-start)
- [Example output](#example-output)
- [How Semantic Similarity Rating works](#how-semantic-similarity-rating-works)
- [Results from the paper](#results-from-the-paper)
- [Limits: read before you trust it](#limits-read-before-you-trust-it)
- [FAQ](#faq)
- [Credits and citation](#credits-and-citation)

## What it does

You describe a product. The skill then:

1. **Creates synthetic consumers:** personas with realistic ages, incomes, jobs and locations for your target market.
2. **Asks each one the survey question separately,** for example "How likely are you to purchase the product?". Each persona gets its own fresh LLM call, so answers don't copy each other.
3. **Scores every answer with SSR.** Each answer becomes a probability for each rating from 1 to 5, not a single guessed number.
4. **Writes a report:**
   - the mean rating;
   - the top-2-box share (people rating 4 or 5);
   - the spread of ratings;
   - results by age or income group;
   - the most negative and most positive answers, which show what puts people off and what sells.

Use it for concept testing, price checks, landing page and ad copy, feature ideas, or picking between two product versions before you spend money on a real survey.

## Why not just ask the AI for a rating?

Ask an LLM "rate 1 to 5" and it plays it safe: nearly every answer is a 3 or a 4. The average can look fine, but the spread is unrealistic. In the paper, this direct method matched the real rating spread with a KS similarity of only **0.26**.

SSR lets the model answer like a person ("I like it, but $29 is a lot for me right now"). It then measures how close that sentence is in meaning to reference sentences for each rating. That keeps the uncertainty in the answer and gives realistic spreads (**0.88**).

## Install

**As a Claude Code plugin:**

```
/plugin marketplace add AryamanSi17/synthetic-consumer-research
/plugin install synthetic-consumer-research@synthetic-consumer-research
```

**Or copy the skill folder:**

```bash
git clone https://github.com/AryamanSi17/synthetic-consumer-research
cp -r synthetic-consumer-research/skills/synthetic-consumer-research ~/.claude/skills/
```

**Embeddings.** Pick one:

- **OpenAI `text-embedding-3-small` (recommended, as in the paper):** set `OPENAI_API_KEY`. No extra package is needed.
- **Free and local:** `pip install sentence-transformers`. It uses `all-MiniLM-L6-v2`.

The scripts are plain Python with no other dependencies. Collecting answers uses the `claude` CLI by default, or any command line LLM tool via `--llm-cmd`.

## Quick start

In Claude Code, just ask:

> Test this product idea on 100 synthetic customers in the US: a $29 cold brew bottle that makes café-style cold brew overnight.

Claude follows the skill: it writes the concept and personas, runs the survey, scores it and explains the results.

Or run it yourself:

```bash
cd skills/synthetic-consumer-research
python3 scripts/simulate.py --concept ../../examples/concept.md --personas ../../examples/personas.jsonl --samples 2
python3 scripts/ssr.py responses.jsonl --by age_band --out results
```

Already have real survey results? Add `--human 12,30,80,150,90` (the counts for ratings 1 to 5) to measure how close the synthetic survey gets. This is the best way to check the setup before trusting it.

## Example output

This is from the included [example](examples/): 8 personas with 2 answers each, scored with the free local model. See the full [report](examples/results/report.md).

```
Mean rating: 3.33 / 5
Top-2-box (rated 4 or 5): 49%

  1  ###                             10.1%
  2  ######                          21.3%
  3  ######                          20.0%
  4  #######                         22.4%
  5  ########                        26.2%
```

The answers are often more useful than the score:

> "On a fixed income, $29 for a bottle is a lot to spend on something I can already make for pennies." (65+, retired)

> "I buy cold brew most days and a bottle that makes it at home for much less is appealing. I'd want to hear from people who've used it, though." (35-44, high income)

This tiny run is only a demonstration. Use 100 or more personas for real decisions. The local model packs scores closer together than OpenAI embeddings do, so use OpenAI when it matters.

## How Semantic Similarity Rating works

For each answer and each set of five reference sentences (one sentence per rating, 1 to 5):

1. **Embed** the answer and the five reference sentences.
2. **Take the cosine similarity** of the answer to each reference sentence: `s1 ... s5`.
3. **Subtract the smallest similarity** from all five. All the reference sentences are about the same topic, so the raw similarities are bunched together. Subtracting the minimum spreads them out.
4. **Normalise** so the five numbers sum to 1. That gives the probability of each rating.

```
p(r) = (s_r − s_min) / Σ (s_k − s_min)
```

The skill averages six different reference sets, which makes the result less dependent on any one wording. Optional temperature sharpens or flattens the result: `p(r)^(1/T)`. The survey result is the average over all answers.

The reference sentences live in [`anchors/`](skills/synthetic-consumer-research/anchors/), with one file for purchase intent and one generic agree/disagree file. Edit them or add your own JSON for other questions.

## Results from the paper

| Method | Matches real concept ranking (correlation attainment) | Matches real rating spread (KS similarity) |
|---|---|---|
| Ask for a number directly | about 80% | 0.26 |
| Free text, then an LLM picks a number | 85% | 0.72 |
| **Free text + SSR** | **about 90%** | **0.88** |

These figures are for GPT-4o, with the same pattern for Gemini 2.0 Flash. Other findings:

- **Demographic personas matter.** Without them, correlation dropped from 92% to 50%.
- **Age and income effects were reproduced well.** Gender and region effects were not.
- **Text descriptions** work almost as well as concept images.

## Limits: read before you trust it

- **This is a first filter, not real customers.** Use it to rank ideas and find objections quickly. Confirm important decisions with real people.
- **Narrow validation.** The method was proven on personal care products in the United States. Other categories, countries and languages are less tested.
- **Training data limits.** The model only knows its training data, so new categories, local brands and cultural habits may be off.
- **Unreliable splits.** Gender and region splits weren't reliable in the paper.
- **Unpublished reference sentences.** The paper's own sentences aren't published. The ones here follow its rules (1 = won't buy, 3 = unsure, 5 = will buy), and results move slightly with the wording.

## FAQ

**What is synthetic consumer research?**
Using AI personas to answer survey questions as customers would, so you can test ideas before paying for a real panel.

**What is a Likert scale?**
The familiar 1 to 5 survey scale, from "definitely would not buy" to "definitely would buy".

**What is Semantic Similarity Rating (SSR)?**
A way to turn free-text answers into Likert ratings by comparing their meaning with reference sentences, using embeddings. It was introduced in arXiv:2510.08338.

**Does it need Claude?**
The skill is written for Claude Code, but the scripts work on their own. `simulate.py --llm-cmd` accepts any LLM command line tool, and `ssr.py` scores answers from any source.

**How much does it cost?**
One short LLM call per answer, plus embeddings that cost a fraction of a cent per 1,000 answers. The local embedding model is free.

**Can I score real open-ended survey answers?**
Yes. Put them in a JSONL file with a `response` field and run `ssr.py`.

## Credits and citation

- **Method:** Benjamin F. Maier, Ulf Aslak, Luca Fiaschi, Nina Rismal, Kemble Fletcher, Christian C. Luhmann, Robbie Dow, Kli Pappas and Thomas V. Wiecki.
- **Official reference implementation:** [pymc-labs/semantic-similarity-rating](https://github.com/pymc-labs/semantic-similarity-rating), Apache 2.0. This skill is an independent, dependency-free version of the same formula, plus persona simulation and reporting.

```bibtex
@article{maier2025ssr,
  title   = {LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings},
  author  = {Maier, Benjamin F. and Aslak, Ulf and Fiaschi, Luca and Rismal, Nina and Fletcher, Kemble and Luhmann, Christian C. and Dow, Robbie and Pappas, Kli and Wiecki, Thomas V.},
  journal = {arXiv preprint arXiv:2510.08338},
  year    = {2025}
}
```

Made by [Aryaman Sinha](https://github.com/AryamanSi17) · Instagram [@ai_aryaman](https://www.instagram.com/ai_aryaman/)
