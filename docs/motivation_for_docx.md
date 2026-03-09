# Motivation — For Technical Draft (Copy into DOCX)

> **Instructions**: Copy the sections below into your "LOATO Bench Technical Draft1.docx"
> where appropriate (e.g., Introduction or Motivation section). Delete this file after use.

---

## Why Input-Layer Prompt Injection Classifiers?

Modern frontier large language models (LLMs) such as Claude (Anthropic), GPT-4 (OpenAI), and
Gemini (Google) have become increasingly robust against known prompt injection attacks through
extensive reinforcement learning from human feedback (RLHF) and safety-focused fine-tuning.
Publicly available adversarial prompts from datasets collected in 2023–2024 — including
HackAPrompt, Open-Prompt-Injection, and Deepset Prompt Injections — are now largely ineffective
when tested against these frontier models.

**This does not mean prompt injection is a solved problem.** Rather, it means the threat landscape
has evolved, and defense strategies must evolve with it. We identify five key reasons why
lightweight, embedding-based input classifiers remain critically important:

### 1. Indirect Injection via RAG Pipelines

The most pressing real-world threat is *indirect prompt injection*, where malicious instructions
are embedded within documents, emails, web pages, or database records that are retrieved and
provided to an LLM as context through Retrieval-Augmented Generation (RAG) pipelines. In this
scenario, the adversarial content arrives as "data" rather than a direct user prompt, which
significantly reduces the effectiveness of the LLM's own conversational guardrails. An input
classifier that inspects all retrieved content *before* it reaches the LLM provides a critical
defense layer that the model itself cannot reliably provide.

### 2. Model-Agnostic Protection

Not all deployed LLMs benefit from the extensive safety training of frontier commercial models.
Open-source models (Llama, Mistral, Phi), custom fine-tuned models, and smaller task-specific
models frequently lack robust injection defenses. A model-agnostic embedding-based classifier
provides consistent protection regardless of the downstream model, making it suitable for
enterprise environments where multiple LLMs may be deployed across different applications.

### 3. Cost and Latency Efficiency

Relying on an LLM to self-police against injection attacks — or routing every input through a
secondary safety-focused LLM call — incurs significant computational cost and latency. A
lightweight embedding classifier (approximately 1ms inference time for models like MiniLM-L6-v2)
is orders of magnitude cheaper and faster, making it practical for high-throughput production
systems where every millisecond and every API dollar matters.

### 4. Defense in Depth

Security best practices dictate that production systems should never rely on a single defensive
layer. Even well-aligned frontier LLMs can be bypassed by sufficiently novel attack techniques.
An independent input classifier provides a complementary detection layer that operates on
different principles (embedding geometry) than the LLM's own safety training (RLHF), reducing
the probability that a single novel technique defeats both defenses simultaneously.

### 5. Generalization to Novel Attack Types (Core Research Contribution)

The central research question of this work — evaluated through the LOATO (Leave-One-Attack-Type-Out)
protocol — directly addresses the most practically important scenario: **can a classifier detect
attack types it has never seen during training?** New prompt injection techniques emerge
continuously and often spread rapidly through the research community and adversarial actors.
A classifier that generalizes across attack categories provides proactive protection rather than
reactive patching.

### Framing

The fact that frontier LLMs now resist previously known injection patterns *strengthens* rather
than weakens the case for this research. It demonstrates that (a) the arms race between attacks
and defenses is ongoing, (b) defenses that rely solely on the LLM's own training will always lag
behind novel techniques, and (c) there is a clear need for lightweight, generalizable,
model-agnostic detection systems that complement LLM-level safety — which is precisely what
LOATO-Bench evaluates.

---

## Suggested Demonstration Approach (for Presentation)

Rather than attempting to demonstrate successful prompt injection against a frontier LLM (which
would likely fail due to current safety guardrails), we recommend the following demonstration
strategy:

1. **Classifier-as-product demo**: Show the classifier detecting and flagging adversarial inputs
   *before* they reach any LLM — this is the actual product/contribution.

2. **RAG pipeline scenario**: Construct a realistic RAG demo where a benign-looking document
   contains embedded injection instructions. Show that (a) the document passes basic content
   filters, but (b) the embedding classifier flags the injected content when it enters the
   retrieval pipeline.

3. **Cross-model vulnerability**: Optionally demonstrate that the same adversarial prompts that
   frontier models now resist *do* succeed against a smaller open-source model, illustrating
   the need for model-agnostic input-layer defense.

4. **LOATO generalization**: The core contribution — show that the classifier maintains detection
   performance on attack categories entirely absent from its training data, demonstrating
   practical utility against the next generation of unknown attacks.
