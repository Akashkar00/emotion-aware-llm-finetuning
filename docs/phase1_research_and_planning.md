# Phase 1 — Research & Planning

## Problem Statement

Modern LLM systems — assistants, support agents, companions, tutoring bots — are
evaluated almost entirely on task completion and factual correctness. They are
largely blind to *how* the user feels while asking. A support bot that responds
to "I've re-installed this three times and it still crashes" with the same tone
it would use for "how do I install this" is missing signal that a human agent
would pick up in one sentence.

This matters for three concrete reasons:

1. **Response calibration.** The right answer to a frustrated user is often the
   same *content* delivered with different framing (acknowledge first, then
   fix). Systems that can't detect the emotional state can't make that choice.
2. **Escalation and safety.** Detecting grief, anger, or fear reliably is a
   prerequisite for routing to a human, suppressing a flippant response, or
   triggering a safety flow — this is already a production requirement in
   health, mental-health-adjacent, and customer-support LLM deployments.
3. **It is a genuinely hard NLP problem**, which is why it's a good vehicle for
   demonstrating fine-tuning skill: emotions are frequently co-occurring
   (multi-label, not multi-class), subtle, imbalanced in natural data, and
   context-dependent — sentiment analysis's easy cousin, but harder.

The engineering claim this project makes is narrow and falsifiable: *a small
open LLM, adapted with parameter-efficient fine-tuning, can match or approach
purpose-built classifier baselines on fine-grained multi-label emotion
detection, at a fraction of the trainable-parameter cost.* Everything from
Phase 4 onward exists to test that claim honestly, including the baselines
that might beat it.

## Literature Review

### From sentiment to emotion

**Sentiment analysis** collapses text into polarity (positive / negative /
neutral) — a 3-way, mutually-exclusive classification problem. It's the
default starting point in most NLP courses because it's cheap to label and
cheap to evaluate, but it throws away almost all affective information: "I'm
furious" and "I'm heartbroken" are both just "negative."

**Emotion classification** predicts a specific emotion (anger, joy, fear,
sadness, ...), typically grounded in a psychological taxonomy. The two
dominant taxonomies in NLP:

- **Ekman's six basic emotions** (anger, disgust, fear, joy, sadness,
  surprise) — compact, cross-culturally validated, but coarse.
- **Plutchik's wheel of emotions** — eight primary emotions arranged with
  intensity and opposition (e.g., joy vs. sadness), which also naturally
  motivates *multi-label* structure since adjacent emotions can co-occur.

**Multi-label classification** is the structural shift that makes this hard:
a single input can carry several true labels simultaneously ("excited but
nervous" → `joy` AND `fear` both true), labels are not mutually exclusive,
and the loss, metrics, and even the output head all change (sigmoid + BCE
per class instead of softmax + cross-entropy). This is the correct framing
for real text, because humans do report mixed emotional states — GoEmotions'
own annotations show a meaningful fraction of examples carry 2+ labels.

### Encoder models (the classification-head lineage)

- **BERT** (Devlin et al., 2018) — bidirectional masked-LM pretraining;
  established the pretrain-then-finetune paradigm for encoder-only models via
  a `[CLS]` token + linear head. Still a reasonable classification baseline;
  110M params (base).
- **RoBERTa** (Liu et al., 2019) — same architecture as BERT, but fixes the
  pretraining recipe: more data, longer training, dynamic masking, dropped
  the next-sentence-prediction objective. Consistently outperforms BERT on
  GLUE-style tasks at the same parameter count, which makes it a stronger
  default baseline than BERT for this project.
- **DeBERTa / DeBERTa-v3** (He et al., 2021) — adds disentangled
  attention (content and position encoded and attended to separately) and,
  in v3, replaces MLM pretraining with **ELECTRA-style replaced-token
  detection**, which is a materially more sample-efficient objective. It is
  the current strongest encoder-only baseline for fine-grained multi-label
  text classification and is the model to beat in Phase 4.

These three represent the "traditional" path: bidirectional encoder →
classification head → fine-tune all weights (or close to it) with a
cross-entropy/BCE loss. Small, fast, cheap to fully fine-tune, no generation
capability.

### Decoder LLMs (the instruction-tuning / PEFT lineage)

- **Llama 3** (Meta, 2024) — decoder-only, dense transformer, strong
  general-purpose instruction-following after chat-tuning; 8B is the smallest
  officially released dense size in the family relevant here. Good base for
  LoRA/QLoRA experiments if VRAM allows (8B in 4-bit ≈ 5–6GB, trainable
  comfortably on a single 16GB+ GPU).
- **Phi-3 Mini** (Microsoft, 2024) — 3.8B, trained on a heavily curated /
  synthetic "textbook-quality" corpus; punches above its parameter count on
  reasoning benchmarks. Attractive here specifically because its small size
  makes QLoRA fine-tuning and later inference-latency benchmarking (Phase 10)
  tractable on modest hardware, while still being a genuine decoder LLM
  rather than a toy.
- **Gemma / Gemma 2** (Google, 2024) — 2B/9B, dense decoder, distilled from
  larger internal models; the 2B variant is the smallest true LLM in this
  comparison and the cheapest to iterate on for hyperparameter sweeps
  (Phase 8), at some cost to ceiling quality.

Framing these as classifiers requires a choice this project takes seriously
in Phase 5: either (a) generate structured output (JSON or a constrained
template) via instruction tuning, or (b) attach a classification head to the
decoder's final hidden state, à la `AutoModelForSequenceClassification` on a
causal LM. Approach (a) is what makes this an *LLM fine-tuning* project
rather than "yet another BERT classifier with extra steps," and it's the one
this project commits to — but it's worth naming that it trades away some
metric cleanliness (generated JSON can be malformed; a classification head
never is) for the skill this project exists to demonstrate.

### Why this project is structurally harder than typical fine-tuning demos

Most public LoRA fine-tuning tutorials target single-label classification or
free-form generation, where "did it work" is easy to eyeball. Multi-label
emotion detection forces engineering rigor that those demos skip:

- The loss function and output parsing are non-trivial for a decoder LLM
  (Phase 5/6).
- Evaluation requires macro-F1, per-class PR curves, and imbalance-aware
  metrics — accuracy alone is actively misleading on a 27-class imbalanced
  set (Phase 9).
- A meaningful ablation (LoRA vs. QLoRA vs. full fine-tuning vs. encoder
  baselines) requires a real experiment harness, not a single training run.

That's the intended signal for a resume: not "I ran a LoRA script," but "I
made the classic classifier-vs-LLM, and PEFT-vs-full-fine-tune, trade-offs
concrete with numbers."

## Why GoEmotions?

**Size:** 58,009 curated Reddit comments — large enough to fine-tune small-
to-mid encoders from scratch and to instruction-tune a PEFT adapter without
needing synthetic data augmentation, small enough to iterate on with modest
compute (full dataset fits comfortably in memory; no distributed data loading
needed).

**Labels:** 27 fine-grained emotion categories + `neutral` (admiration,
amusement, anger, annoyance, approval, caring, confusion, curiosity, desire,
disappointment, disapproval, disgust, embarrassment, excitement, fear,
gratitude, grief, joy, love, nervousness, optimism, pride, realization,
relief, remorse, sadness, surprise), each example multi-label (~1.2 labels/
example on average per the original paper), with an official secondary
mapping down to Ekman's 6 + neutral and to 3-way sentiment. That gives this
project three difficulty tiers "for free": fine-grained (27-way), Ekman
(7-way), and sentiment (3-way) — useful for framing results at different
granularities in Phase 9 rather than only reporting on the hardest setting.

**Challenges (the reason this dataset is a legitimate research vehicle, not
just a convenient one):**

- **Label imbalance.** Frequent classes (`admiration`, `approval`, `neutral`)
  have thousands of examples; rare classes (`grief`, `pride`, `relief`) have
  only on the order of a hundred. Macro-F1 will be dominated by how well the
  model does on classes it barely sees — this is precisely the failure mode
  naive fine-tuning glosses over, and precisely what Phase 2's imbalance
  analysis and Phase 9's per-class evaluation exist to surface.
- **Multi-label ≠ multi-class.** Requires sigmoid outputs, per-class
  thresholds (not argmax), and multi-label-aware metrics throughout the
  pipeline — a genuine implementation detail, not a config flag.
- **Label noise / subjectivity.** Reddit comments were labeled by 3+ raters
  with documented inter-rater agreement well below 1.0 for adjacent emotions
  (e.g., `annoyance` vs. `anger`, `desire` vs. `optimism`); some ceiling on
  achievable F1 is inherent to the data, not the model, and the research
  report (Phase 12) should say so explicitly rather than presenting F1 as if
  100% were attainable.
- **Domain-specific language.** Reddit-style text (slang, sarcasm, ALL CAPS
  emphasis, emoji-adjacent punctuation) diverges from the clean-text domains
  most base LLMs were most heavily pretrained/instruction-tuned on, so
  domain adaptation is doing real work here, not just syntax memorization.

Source: Demszky et al., *"GoEmotions: A Dataset of Fine-Grained Emotions"*
(2020), [arXiv:2005.00547](https://arxiv.org/abs/2005.00547); dataset and
card at the [google-research/goemotions](https://github.com/google-research/google-research/tree/master/goemotions) repo.

## Model Selection Analysis

| Model | Params | Type | Full FT feasible? | PEFT-friendly? | Fit for this project |
|---|---|---|---|---|---|
| DistilBERT | 66M | Encoder | Yes, trivially | N/A (usually fully fine-tuned) | Good *floor* baseline — cheapest, sets the "what does almost no effort buy you" reference point |
| BERT-base | 110M | Encoder | Yes | N/A | Standard baseline; mainly useful as the historical reference point everything else is implicitly compared to |
| RoBERTa-base | 125M | Encoder | Yes | N/A | Stronger baseline than BERT at near-identical cost; expected to be competitive in Phase 4 |
| DeBERTa-v3-base | 184M | Encoder | Yes | N/A | Expected strongest encoder baseline — the one the LLM+PEFT approach actually needs to be compared against honestly |
| Phi-3 Mini | 3.8B | Decoder LLM | No (impractical on consumer HW) | Yes — LoRA/QLoRA both comfortable on a single 16GB GPU | **Primary candidate.** Best size/capability trade-off for iterating through Phases 6–10 on modest hardware |
| Gemma 2B | 2.6B | Decoder LLM | Marginal | Yes, cheapest of the three LLMs to sweep | Best choice if compute is the binding constraint (e.g., free-tier Colab); good for Phase 8's hyperparameter sweep specifically, even if Phi-3 Mini is used for the "main" run |
| Llama 3 8B | 8B | Decoder LLM | No | Yes via QLoRA (4-bit) on a single 16–24GB GPU | Best ceiling if GPU budget allows; use as the "if resources permit" upgrade path, not the default |

**Recommendation:**

- **Baselines (Phase 4):** run all of DistilBERT / BERT-base / RoBERTa-base /
  DeBERTa-v3-base — they're cheap, and having all four makes the comparison
  table honest instead of cherry-picked.
- **LoRA/QLoRA target (Phases 6–7):** **Phi-3 Mini** as the primary model.
  Reasoning: it's the best-documented mid-size decoder for PEFT tutorials
  right now, small enough that QLoRA fine-tuning and the hyperparameter
  sweep in Phase 8 are actually iterable in a single session rather than a
  multi-hour-per-run bottleneck, and still a real instruction-tuned decoder
  LLM (not a toy) — which matters for the resume claim.
- **Stretch goal:** if GPU budget allows later, repeat the best-config LoRA
  run on **Llama 3 8B** via QLoRA as a "does the finding hold at larger
  scale" check — this is exactly the kind of extra ablation that separates a
  strong portfolio project from a tutorial clone, but it should be optional
  and explicitly labeled as a stretch item so scope doesn't creep before the
  core pipeline (Phases 2–10) is done.
- **Not recommended as primary:** Gemma 2B as the *main* reported model —
  it's the right choice only if compute is severely constrained; keep it in
  reserve for Phase 8's cheaper sweep iterations rather than the headline
  run, so the headline number comes from the model most people will actually
  ask about in an interview (Phi-3 or Llama 3).

---

**Next up (Phase 2):** full EDA on GoEmotions — dataset statistics, class
distribution, label-imbalance analysis, multi-label distribution, and
correlation between emotions, with visualizations.

Ready for the next phase?
