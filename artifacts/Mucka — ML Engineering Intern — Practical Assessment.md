# Mucka — ML Engineering Intern

## Practical assessment: triage the inbox

**Timebox: 3 hours.** Read this whole page before you start.  
th  
---

## The situation

Mucka builds an AI business operating system for UK trades businesses — plumbers, gas engineers & HVAC. Our users are owners and office managers, not technical people.

One of the firms we work with has 14 engineers and takes 60–120 inbound messages a day, arriving as SMS, WhatsApp, and transcribed voicemail. At the moment the office manager reads every single one, in the order they arrive, and decides what happens next. When it gets busy, things get missed.

## Your task

**Build something that decides, for each inbound message, how urgently a human needs to see it.** Then give us an honest assessment of whether we should put it in front of a customer.

That is deliberately the entire brief. A lot has not been specified: what the urgency levels should be, what "good" means, how to measure it, what method to use. Those decisions are yours. Making them sensibly, and telling us why, is most of what we are assessing.

## The data

Two Google Sheets. Use **File → Download → Comma-separated values** to get CSVs.

**messages** — 1,252 inbound messages. Some have a value in the `urgency` column; most do not. [https://docs.google.com/spreadsheets/d/1shudyDL7hSa86zm\_JAv6ddUFg2fVm285OMGIBi4YNj4/edit](https://docs.google.com/spreadsheets/d/1shudyDL7hSa86zm_JAv6ddUFg2fVm285OMGIBi4YNj4/edit)

**holdout** — 300 further messages. No labels. [https://docs.google.com/spreadsheets/d/1HruuVQDQYBFLXlydBfvFc3M\_LHrJNfjwI-5ISa7jvQc/edit](https://docs.google.com/spreadsheets/d/1HruuVQDQYBFLXlydBfvFc3M_LHrJNfjwI-5ISa7jvQc/edit)

Columns: `message_id`, `received_at`, `channel` (sms / whatsapp / voicemail), `from_name`, `body`, and `urgency` (messages only).

The data is synthetic, but it is modelled closely on the real thing. Treat it exactly as you would data pulled out of a live system — the messy parts are the realistic parts.

## What to hand in

A git repo — a GitHub link, or a zip — containing:

1. **Code we can run.** A `README.md` that tells us how, starting from a clean checkout.  
2. **`predictions.csv`** — your system's output for every row in the holdout set. At minimum `message_id` and your predicted urgency; add columns if they're useful.  
3. **Your evaluation** — the numbers, and the code that produced them. You decide what to measure. Tell us why you measured that.  
4. **`memo.md`** — one page, written to the founder, who is not technical. Would you ship this? What would you need in order to be confident? Write it for someone who runs a plumbing business, not for an ML team.

## Rules

- **2.5 hours.** Stop when you hit 2.5 hours, then add a short "what I'd do with more time" section to the README. Finishing early is fine and is not penalised.  
- **AI coding assistants are allowed and expected.** You must be able to explain every line you hand in.  
- Any libraries, any approach, internet open.  
- If you want to call a hosted LLM API and don't have a key, say so and we'll provide one. Nothing in this task requires one.  
- **Questions:** [tom.webster@mucka.ai](mailto:tom.webster@mucka.ai).

## How we'll assess it

We are not counting your accuracy score. We are reading for three things:

- Whether your evaluation would survive contact with reality.  
- Whether you noticed the things we didn't tell you about.  
- Whether that memo would actually help a busy person make a decision.

Afterwards we'll sit down for about half an hour and you can walk us through what you built and what you'd do next.

Good luck.  
