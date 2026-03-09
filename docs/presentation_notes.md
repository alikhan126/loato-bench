# LOATO-Bench — Capstone Presentation Notes

> **Format**: 7 slides, ~5-7 minutes. Each section = one slide.
> Bullet points = what goes on the slide. Speaker notes = what you say out loud.

---

## Slide 1: Title

**LOATO-Bench**
*Can Prompt Injection Classifiers Detect Attacks They've Never Seen Before?*

Ali Khan, Ahmad Mukhtar — MS Data Science Capstone, Pace University

**Speaker notes:**
Hi everyone, my capstone project is called LOATO-Bench. The core question I'm trying to answer is simple: if we train a classifier to detect certain types of prompt injection attacks, will it also catch completely new types of attacks it's never been trained on? That's what this project is all about.

---

## Slide 2: What's the Problem?

- AI chatbots and assistants (ChatGPT, Claude, etc.) can be tricked with specially crafted inputs — this is called "prompt injection"
- The big models from OpenAI and Anthropic have gotten better at blocking known tricks
- But new tricks keep appearing — it's an arms race
- Smaller and open-source AI models have almost no protection at all
- The biggest risk today: attackers hide malicious instructions inside normal-looking documents that get fed to AI systems (through RAG pipelines)
- **We tested this**: Both Claude Sonnet and GPT-4o-mini were compromised on 3/5 RAG-style indirect injection tests (see `docs/llm_vulnerability_demo.md`)

**Speaker notes:**
So what's the problem? You've probably heard of prompt injection — it's when someone crafts a sneaky input to make an AI do something it shouldn't. Now, the big commercial models like GPT-4 and Claude have gotten pretty good at blocking the known tricks. But here's the thing — new attacks keep popping up all the time. And if you're using a smaller model, or a fine-tuned open-source model, they often have zero protection. The scariest part is what's called indirect injection — imagine a company uses AI to summarize documents. An attacker hides instructions inside a normal-looking PDF, and when the AI reads it, it follows those hidden instructions instead of doing its job. And this isn't hypothetical — we actually tested it. We sent 5 different injection attacks hidden inside normal-looking documents to both Claude and GPT-4o, and both models were compromised on 3 out of 5 tests. They even failed on different attacks, which means no single vendor has solved this. That's a real, demonstrable threat right now.

---

## Slide 3: Why Not Just Let the AI Protect Itself?

- Relying on the AI model alone is like having only one lock on your door
- A lightweight classifier can screen inputs in ~1 millisecond — way cheaper and faster than running another AI call
- Works with ANY AI model underneath (model-agnostic)
- Adds a separate layer of defense — even if the AI's own safety fails, this catches it first
- **Our demo proves it**: Classifier blocked 5/5 poisoned documents that fooled both frontier LLMs

**Speaker notes:**
You might ask — if the big models are getting better at blocking attacks, why do we even need a separate classifier? Think of it like home security. You wouldn't just rely on one lock. You'd want a deadbolt, maybe a camera, an alarm. Same idea here. An embedding-based classifier is like a fast security scanner at the door. It checks every input in about a millisecond — that's basically instant — and it works regardless of which AI model you're using behind it. So even if the AI's own defenses fail against some new trick, this classifier is there as a backup. And it's dirt cheap to run compared to making another AI call for safety checking. In our demo, the classifier blocked all 5 poisoned documents — including the 3 that each LLM actually fell for. That's the value of defense in depth.

---

## Slide 4: What's Missing in Current Research?

- Existing studies test classifiers the easy way: train and test on the same types of attacks
- That's like studying for a test with the answer key — of course you'll do well
- Nobody has systematically asked: "What happens when you encounter a completely new type of attack?"
- That's the most important question for real-world use

**Speaker notes:**
Here's what got me excited about this project. When I looked at the existing research on prompt injection classifiers, I noticed something. Everyone evaluates their classifiers using standard cross-validation — basically, you shuffle your data, split it into train and test, and measure performance. But the train and test sets contain the same types of attacks. That's like studying for an exam where you already know which topics will be covered. Of course you'll score well. But in the real world, attackers don't use the same tricks you trained on. They come up with new ones. So the question that really matters is: will your classifier still work when it sees something completely new? And nobody had systematically tested that. That's the gap this project fills.

---

## Slide 5: My Approach — LOATO

- **LOATO** = Leave-One-Attack-Type-Out
- I categorized prompt injections into 8 types (e.g., "ignore previous instructions", roleplay jailbreaks, encoded attacks, hidden instructions in documents, etc.)
- Training: use 7 out of 8 attack types
- Testing: see if the classifier catches the one type it's never seen
- Repeat for each type — this tells us how well the classifier truly generalizes
- I measure the "generalization gap" — how much does performance drop on unseen attacks?

**Speaker notes:**
So here's my approach. I call it LOATO — Leave One Attack Type Out. I've organized prompt injection attacks into 8 categories. Things like "ignore your previous instructions", jailbreak roleplay where you tell the AI to pretend it has no rules, encoded attacks using Base64 or other tricks, hidden instructions embedded in documents, and so on. The evaluation works like this: I train the classifier on 7 of these 8 categories, and then test it on the one category it has never seen. Then I rotate — leave out a different category each time. This tells me exactly how well the classifier generalizes to truly novel attacks. I also compare this against standard evaluation to measure the "generalization gap" — basically, how much does performance drop when you encounter something new? A small gap means real generalization. A big gap means the classifier was just memorizing patterns.

---

## Slide 6: What I've Built So Far

- **Unified dataset**: Combined 5 public prompt injection datasets into one benchmark (~20,000+ samples)
  - Cleaned duplicates, detected languages, validated quality
- **Attack taxonomy**: Built a 3-tier system to categorize every sample into one of 8 attack types (~70% mapped so far, rest coming via LLM-assisted labeling)
- **Full EDA**: Validated data quality, identified filtering needs, confirmed experiments are feasible
- **Embedding pipeline**: 5 different embedding models ready to go (from lightweight to heavyweight)
- **Code quality**: 107 tests passing, 90%+ coverage, CI pipeline, type checking — production-grade infrastructure
- **Next steps**: Finalize taxonomy → build classifiers → run the full experiment matrix

**Speaker notes:**
Let me show you where I am. I've built the full data infrastructure. I took 5 publicly available prompt injection datasets — about 20,000+ samples total — and unified them into one clean benchmark. That involved deduplication, language detection, and quality validation. One dataset alone had 177,000 samples but most of them were mislabeled, so I built a quality gate to filter down to genuine injection samples. I created a taxonomy system that categorizes every sample into one of 8 attack types — about 70% are mapped so far, and the rest will be handled by LLM-assisted classification in the next sprint. I've also implemented 5 different embedding models, from a tiny fast one to a large 7-billion parameter one. The codebase has 107 tests passing with 90%+ coverage and a full CI pipeline. So this isn't just a proposal — I have working infrastructure. Next steps are finalizing the taxonomy, implementing the classifiers, and running the full experiment matrix.

---

## Slide 7: Why This Matters + Questions

- **First** systematic study testing prompt injection classifiers on attack types they've never seen
- **Practical outcome**: which embedding + classifier combo works best for real-world deployment
- **Useful for anyone** building AI-powered applications with document retrieval (RAG)
- Benchmark and code will be available for future research

*Questions?*

**Speaker notes:**
To wrap up — this is the first systematic study that tests whether prompt injection classifiers can actually generalize to new attack types. The practical outcome will be clear recommendations: which combination of embedding model and classifier gives you the best protection against unknown attacks, and at what cost. This is directly useful for anyone building AI applications that pull in external documents — which is basically every enterprise AI deployment right now. The benchmark and code will be open for future research to build on. I'm happy to take any questions.

---

## Quick Reference — If Asked

**Q: How many experiments will you run?**
About 270-310 total runs across 4 experiment types: standard cross-validation (baseline), LOATO (primary), direct-to-indirect transfer, and cross-lingual transfer.

**Q: What embedding models?**
Five, ranging from a small fast model (MiniLM, 384 dimensions) to a large 7B model (E5-Mistral, 4096 dimensions), plus BGE, Instructor, and OpenAI's embedding model.

**Q: What classifiers?**
Four: Logistic Regression, SVM, XGBoost, and a neural network (MLP). Standard but covers linear, kernel-based, tree-based, and neural approaches.

**Q: Timeline?**
Infrastructure done. Taxonomy finalization and classifier implementation next, then experiments, analysis, and write-up.

**Q: What if the classifier doesn't generalize?**
That's actually a useful finding too — it would mean current embedding-based approaches are brittle against novel attacks, which is important for the community to know. But I expect some models to generalize better than others, and identifying which ones is the contribution.

**Q: Have you actually tested whether LLMs are vulnerable?**
Yes. We ran a live demo (Scenario 0 in the demo notebook, documented in `docs/llm_vulnerability_demo.md`). We tested both Claude Sonnet and GPT-4o-mini against 5 direct attacks and 5 RAG-style indirect injection attacks. Key findings:
- Direct attacks: both models mostly resist (GPT refused 4/5, Claude deflected 3/5)
- RAG indirect attacks: both models compromised on 3/5 tests (60% success rate for the attacker)
- They fail on *different* attacks — Claude leaked its system prompt, GPT followed an instruction hijack — proving no single vendor has solved this
- Our embedding classifier blocked all 5 poisoned documents, catching every attack both LLMs missed

**Q: What kinds of indirect injection attacks did you test?**
Five types: canary word insertion (hidden "include this word" instruction), response override (replace the answer entirely), instruction hijack (leak system prompt), format manipulation (force French output), and data exfiltration (repeat system message). The subtler attacks succeeded more often — both LLMs fell for canary insertion and format manipulation, while the crude "respond with this exact sentence" override was resisted by both.
