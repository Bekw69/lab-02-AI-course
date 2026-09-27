# Lab 02 — Open the Box: Report

**Course:** AI course, Narxoz University
**Model tested:** GPT-2 small (124M parameters, 2019)

## Introduction

This lab tested three claims from Lecture 3 against a real model instead of just trusting the slides:
(1) the model's output is not a word but a list of 50,257 probabilities, (2) temperature reshapes
that list while top-p deletes part of it, and (3) attention is a table of weights where every row
sums to 1. Below, each prediction is compared against what the model actually did, with an
explanation of why the result came out that way.

---

## Part 0 — Tokenizer size (Kazakh word `бөлімшеңізде`)

**Prediction:** I expected around 18–20 tokens, reasoning that GPT-2's vocabulary (50,257) is the
smallest of the tokenizers compared (cl100k: 100,277, o200k: 200,019), so it should need *more*
tokens per word than both — more than the 12 tokens cl100k used.

**Result:** 19 tokens.

**Explanation:** The prediction was essentially correct. Smaller vocabulary means fewer common
multi-character "chunks" are available, so rare words (especially in scripts the model barely saw
during training) get split much finer. `бөлімшеңізде` split into single UTF-8 bytes rather than
into meaningful sub-word pieces — this connects to the byte-pair merge mechanism: merges are learned
from how often character pairs *co-occur* in training data. GPT-2 trained mostly on 2019 English
internet text, so Cyrillic byte-pairs simply never appeared often enough to earn a merge.

---

## Part 1 — The output is 50,257 numbers

**Prompt:** `The capital of Kazakhstan is Astana. The capital of France is`

**Prediction:** I expected `' Paris'` to be the top token, with roughly 40–60% probability, since
the sentence sets up an easy parallel pattern to complete. I also expected the top-10 tokens to
hold more than 90% of the total probability, given how strong that pattern looked.

**Result:** The actual top token was `' Ast'` (22.87%), with `' Paris'` a close second (22.08%).
The top-10 tokens together held only 64.0% of the probability mass.

**Explanation:** The prediction was partly wrong. The model was not "answering the question" —
it was continuing a *pattern*: since "Astana" had just appeared as a completion for the first half
of the sentence, the model leaned toward repeating that same structure for the second half. This
is the core lesson of the lab: the model's "answer" is the statistically most likely continuation
of the text, not a verified fact. The relatively low top-10 mass (64%, not >90%) also shows the
model was less confident than expected — it was genuinely splitting its "belief" across several
plausible next words. As a side note, `' Paris'` and `'Paris'` (no leading space) got wildly
different probabilities (0.2208 vs 0.000009) because GPT-2's tokenizer treats the leading space
as part of the token — the two are literally different vocabulary entries, and the no-space
version almost never appears in normal running text.

---

## Temperature

**Prediction:** I expected the #1 token to stay the same across T = 0.25, 1, 2, 5, since dividing
every logit by the same positive number shouldn't change their relative order. I expected the
probability of the #1 token to drop sharply as T increased.

**Result:** The #1 token stayed `' Ast'` at every temperature tested (T=0.25 → 53.45%, T=1 → 22.87%,
T=2 → 1.23%, T=5 → 0.04%). Entropy rose from 0.700 nats at T=0.25 to 10.547 nats at T=5, close to
the theoretical maximum of 10.825 nats for a uniform distribution over 50,257 tokens.

**Explanation:** The prediction was correct on both counts. Dividing every logit by the same
constant is a monotonic transformation — it changes the *gaps* between scores but never the
*ranking*, so the argmax token can never move. This also shows why "set temperature to 0 for a
reliable model" is a misleading claim: T=0 always deterministically picks the top token, but that
token (`' Ast'`) was not the factually correct completion of "the capital of France is" — it was
just the model's most confident guess. Reliable and correct are not the same thing.

---

## Temperature vs Top-p (one sentence each)

- **Temperature** reshapes the probability distribution by stretching or compressing the gaps
  between logits — no token's probability is ever forced to exactly zero.
- **Top-p** deletes the tail outright — it keeps only the smallest set of top tokens whose combined
  probability reaches *p*, and sets everything else to exactly zero before renormalizing.

The difference matters at the extremes: at low temperature, top-p only needed to keep 2 tokens to
reach 90% mass, but at T=5 (a nearly flat distribution) it had to keep 35,021 tokens just to reach
the same 90% — proving that "reshape" and "delete" are fundamentally different operations, not two
names for the same thing.

---

## Attention heads

- **Previous-token head:** layer 4, head 11 — this head puts 100% of its attention on the token
  immediately before the current one, producing a clean diagonal one step below the main diagonal
  in the heatmap (row *i* looks at column *i−1*).
- **Most extreme "token 0" (attention-sink) head:** layer 7, head 10 — this head puts 96.4% of its
  attention on the very first token of the prompt, regardless of what that token means semantically.

**Explanation:** Layers 5–10 showed the strongest average attention to token 0 (peaking at ~81% in
layer 7), a pattern known as an "attention sink." This is best explained as a mechanical side effect
of softmax attention: every row of the attention matrix must sum to exactly 1, so a head that has
nothing useful to attend to for a given query is forced to dump its leftover weight somewhere —
and the always-present first token becomes the default destination. This is not evidence that "the
model is thinking about the first word"; it is an artifact of how the mechanism is built. A high
attention weight also only says how much of a token's value vector gets mixed in, not how much that
token actually influenced the model's output — the two are frequently confused but are not the same
thing.

---

## Part 3 — The same model, in Kazakh

**Prediction:** Based on Lecture 3's numbers (Kazakh at 1.7–4.8× English tokens, Russian at
1.2–2.9× across six tokenizers), I expected Kazakh to come out clearly worse than Russian on GPT-2
as well — maybe 20–30% more tokens per character.

**Result:** Russian and Kazakh came out almost identical: 1.11 tokens/char for Russian vs.
1.10 tokens/char for Kazakh (3.88× vs. 4.25× English token count respectively).

**Explanation:** The prediction was wrong on the "Kazakh is clearly worse" point — per-character,
the two languages are essentially tied. What actually breaks is *both* non-English, non-Latin-script
languages, not Kazakh specifically. For both prompts, the model's #1 next-token prediction was
dominated by a broken replacement character (`' \ufffd'`, 37.79% for Russian and 22.54% for Kazakh),
and greedy generation degraded into repetition or nonsense ("Столица Казахста" looping, or "проссии
проссии про" for Kazakh). This confirms Lecture 3's core claim: the problem lives in the
**tokenizer's table**, not in the language itself — GPT-2's vocabulary was built almost entirely
from English text and never learned proper multi-character merges for Cyrillic, so both Russian and
Kazakh get shredded into near-meaningless raw bytes. The fix would not target "Kazakh" specifically;
it would mean retraining (or extending) the tokenizer on a corpus that includes real Cyrillic text
so common words earn proper merges instead of falling back to single bytes.

---

## Conclusion

All three claims from Lecture 3 held up under testing: the model outputs a full probability
distribution rather than a single word; temperature and top-p are mechanically different operations
(reshaping vs. deleting); and attention weights are structurally constrained (rows sum to 1),
producing patterns like attention sinks that are easy to over-interpret as "meaning" when they are
really just artifacts of the softmax mechanism.
