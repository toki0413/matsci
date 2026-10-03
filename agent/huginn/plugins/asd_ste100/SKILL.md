---
name: asd-ste100
description: "Use when English text must be parsed without a human to resolve ambiguity — tool descriptions, error messages, inter-agent instructions, system prompts, status reports — and misreading has a real cost, or when text reads as dense, hedged, or easy to misparse. Triggers: disambiguate, STE100 rewrite, apply Simplified Technical English, plain-language rewrite, controlled-language rewrite, rewrite so an agent cannot misread this. Not for creative or marketing copy."
version: 1.1.0
---

# Simplified Technical English (ASD-STE100)

ASD-STE100 is a controlled-language standard built by the aerospace and defense
industry to stop maintenance technicians from misreading English instructions.
It removes the two biggest sources of misreading: words with more than one
meaning, and sentences with more than one possible structure.

This skill borrows that discipline for a different reader: an AI agent or a
downstream system that must parse an English string — an error message, a tool
description, an inter-agent instruction, a status report — with no human in the
loop to resolve ambiguity. If a technician can misread "close the valve" as an
adjective instead of a command, so can a language model.

## How Huginn Uses This

The plugin `asd_ste100` mounts three live surfaces:

- **Prompt segment** — `ste_prompt_segment` registers into
  `huginn.plugins.prompt_segments`, the registry `build_prompt` assembles from.
  Every system prompt gets the STE rules for the active mode.
- **`ste_lint` tool** — a deterministic, model-free check the agent can run on
  its own draft text before sending it. Rules: sentence length, semicolons,
  phrasal verbs, nominalization, marketing adjectives, synonym rotation, and
  dangling list conjunctions.
- **`comms_lint` tool + an advisory audit on the event bus** — the same
  discipline applied to structured communication (see below).

Mode comes from `HUGINN_STE_MODE` (`agents` | `strict` | `flavored` | `off`,
default `agents`). The feature flag `asd_ste100` is the master switch.

## Structured Communication

STE's three rules — one word one meaning, one message one purpose, explicit
with no back-channel — are not only about prose. They apply to every message
that crosses a module, workflow, or agent boundary. `huginn/comms/contract.py`
turns them into deterministic checks over the shapes Huginn already uses:

- **Event envelope** (`AgentEvent`) — `type`, `timestamp`, and `source` must be
  present; `type` must be a lowercase dotted namespace.
- **Event vocabulary** — a `type` that is not registered (`KNOWN_EVENT_TYPES`)
  is reported, so two names for one event cannot drift apart silently.
- **Payload contract** — events with a declared field set (for example
  `decision.point`, `agent.step.retrying`, `campaign.control_trace`) must carry
  their required fields.
- **Field-name aliases** — `tool` for `tool_name`, `sid` for `session_id`,
  `src` for `source` are reported as drift.
- **Result self-consistency** (`SubagentResult`) — success with an error, or a
  success whose summary is empty, is reported.

The audit is **advisory**: it counts violations and warns once per rule. It
never blocks or drops an event. The feature flag `comms_contract` is its switch.

## When to Use This Skill

- An agent's output (explanation, instruction, log message, tool description)
  reads as dense, jargon-heavy, or ambiguous.
- Text will be consumed by another agent, a translation pipeline, or a
  non-native English reader, and misparsing has a real cost.
- You are writing a prompt, system message, or tool description and want to
  remove ambiguity before a model ever sees it.

Do not apply this skill to creative or marketing copy. STE is deliberately flat
and literal.

## Two Modes

**Strict** — procedures, error messages, tool and function descriptions,
inter-agent instructions, safety text. Apply every rule, including the hard
length caps and the one-word-one-meaning discipline.

**STE-flavored** — READMEs, PR descriptions, changelogs, explanatory prose.
Apply the structural rules in full and treat the lexical rules as advisory.
Prose needs some range, and a strict rewrite of prose reads as a personality
transplant rather than a clarification.

## Core Rewrite Rules

**Structural rules — apply these**

| Rule | Do | Don't |
|---|---|---|
| Active voice | "The agent deletes the file." | "The file is deleted (by the agent)." — unless the actor is unknown or irrelevant |
| No phrasal verbs (Rule 9.3) | "Remove the panel." / "Start the job." | "Take off the panel." / "Spin up the job." |
| One instruction per sentence | "Open the file. Read line 3." | "Open the file and read line 3, then check if it matches." |
| Sentence length | ≤20 words for instructions, ≤25 words for descriptions | Long compound or subordinate-clause sentences |
| No semicolons (Rule 8.1) | Split into separate sentences | Any semicolon at all |
| Noun clusters | ≤3 words stacked as a noun phrase ("fuel pump valve") | 4+ word noun stacks |
| No ellipsis | Keep subject, verb, and article explicit | Drop words to save space |
| Keep modality | "The request may have failed." stays "may have" | Promote a hedge to a fact |
| Paragraph limits | One topic per paragraph, ≤6 sentences | Multi-topic paragraphs |
| Lists for sequences | Use a numbered or bulleted list for 3+ steps | Bury a sequence inside one prose sentence |

**Lexical rules — direction of travel only.** The official ~900-word dictionary
is not reproduced here (ASD does not permit redistribution). Which word is
*approved* is a dictionary fact we cannot verify; preferring the plain word is
safe everywhere.

| Rule | Do | Don't |
|---|---|---|
| One word, one meaning | Pick one verb for one action and reuse it | Rotate synonyms for the same idea |
| Verb, not noun (Rule 3.7) | "Analyze the log." | "Perform an analysis of the log." |
| Domain terms | Keep necessary technical terms; define each once | Use jargon without ever defining it |
| Marketing adjectives | Delete, or state the measurement | "seamless", "robust", "cutting-edge" |

**Simple tenses.** STE permits infinitive, imperative, simple present, simple
past, simple future, and past participle as an adjective. It excludes present
perfect and other compound forms: "we received the report", not "we have
received the report". Where the compound form carries information the simple
form cannot — current relevance, or a hedge as in "may have failed" — keep it
and flag the departure.

## Scan Checklist

1. **Synonym rotation** — the same thing gets several names. Pick one name.
2. **Hedge stacking** — qualifiers pile up until the sentence asserts nothing. State the claim or delete it.
3. **Nominalization** — an action frozen into a noun. Use the verb.
4. **Marketing adjectives** — words that claim quality instead of showing it. Delete, or give the measurement.
5. **Run-on sentences** — several ideas joined by semicolons or em dashes. One idea per sentence.
6. **Soft phrasal verbs** — spin up, reach out, dive into. Use the single plain verb.

## Process

1. Pick the mode (Strict or STE-flavored).
2. Read the input once for meaning. Do not rewrite before you know what it must still say.
3. Flag every violation, sentence by sentence. For a mechanical first pass on
   the structural rules, run `ste_lint` (the tool or
   `python -m huginn.plugins.asd_ste100.ste_lint`).
4. Rewrite each flagged sentence without dropping any fact, condition, or scope
   qualifier. Check modality before you commit: "may have failed" is not
   "failed". Never add a fact the source did not state.
5. If a rewrite would lose required precision, keep the longer phrasing and
   flag the trade-off instead of silently simplifying.
6. If the input already complies, say so. Do not force changes onto compliant text.

## Output Format

Default: the rewritten text, and nothing else. The one permitted addition is a
single line after the text, prefixed `Kept as-is:`, naming a phrase you left
unsimplified and the precision that would have been lost.

On request ("show the diff", "which rules did it break"), output a table:

| Rule violated | Original | Simplified |
|---|---|---|
| Present perfect tense | "We have received your request." | "We received your request." |
| Noun cluster (4+ words) | "the agent task queue priority handler" | "the handler that sets task-queue priority" |

## Boundaries

- Preserve every fact, condition, and scope qualifier.
- Preserve the strength of every hedge. Add no claim the source did not make.
- Do not claim aerospace-grade compliance. This is a clarity tool inspired by
  STE, not a certified STE authoring tool, and it does not reproduce ASD's
  official dictionary.
- STE fixes the form of a text, not its substance. A hollow paragraph becomes a
  clean, short, well-punctuated hollow paragraph.
- Stop when the sentence is unambiguous, not when it is shortest.

## Sources

- [ASD-STE100 official site](https://www.asd-ste100.org/)
- [Simplified Technical English — Wikipedia](https://en.wikipedia.org/wiki/Simplified_Technical_English)

This plugin is a port of [danyuchn/asd-ste100-skill](https://github.com/danyuchn/asd-ste100-skill)
(MIT). The linter is a faithful port of that project's `scripts/ste-lint.py`.