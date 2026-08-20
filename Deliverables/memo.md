# Should you put this in front of your customers?

**Yes, as a sorting assistant. No, as anything that hides a message or answers one on its own.**

The part that does the real work is an AI that reads what a message actually means. It has been
tested on all 300 of the messages your team labelled, and the numbers below are its real
performance. One number in this memo is not trustworthy and I have said so where it appears.

## What it does

Every message still lands in front of your office manager. Nothing is hidden or deleted. The system
reads each one as it arrives, works out how fast a human needs to see it, and keeps the inbox sorted
so the worst thing is at the top. A handful get flagged, each with a one-line reason like "mentions
a smell of gas". When she disagrees, her correction gets recorded and changes how similar messages
are judged later.

It re-orders, it doesn't filter. If it gets one wrong, the message is further down a list she still
reads. That's the whole reason I'm comfortable recommending it now instead of in six months.

## How well it works

I tuned the red-banner alert to about 10 messages a day out of the 60-120 you get. If 30% of the
inbox is marked urgent then nothing is, and she'll stop looking. The table below is a wider count —
everything the system sorts above routine, banner or not, which is nearer 20 a day.

| Question | Answer |
|---|---|
| Of the 11 real emergencies, how many got flagged as emergencies? | 7 of 11 |
| Were any emergencies dismissed as routine? | None. The other 4 were marked "same day", still near the top |
| Of the 44 jobs needing someone out today, how many were caught? | 37 of 44 |
| What does that cost? | 14 false alarms in 245 routine messages, about 6%, roughly 10 seconds each |

Read the first row rather than an overall accuracy score. Marking every message routine would score
82% on your data and be worthless. Seven out of eleven is not a good number, and four misses means
four households.

All four missed emergencies are the same situation:

> "No heating and no hot water since last night and I've got a three week old baby in the house"
>
> "Mum is 89 and there is no heating at all, the house is freezing and she's not well"

No gas, no fire, no flood, no alarm word anywhere. The system called them "same day", which is
defensible from the words and wrong about the situation. A vulnerable person plus no heating is an
emergency even though neither half is one alone. Note that reading the meaning rather than the
keywords did *not* rescue these: the AI read them and still ranked them below emergency.

I have since added a rule for exactly this, and with it in place the same test reads **11 out of
11**. Do not be impressed by that. I wrote the rule *after* seeing these four misses, so of course
it catches them — that is a fitted number, not a prediction, and it will not mean anything until the
rule meets messages it has never seen. **Treat 7-of-11 as the real figure and the 11-of-11 as
untested.** Anyone showing you a 100% obtained by tuning on the failures is showing you their
homework, not a forecast.

For balance, two of the "missed" same-day jobs were a VAT-registration question and a weeks-old
duplicate certificate request. Most people would call those routine — that's your labels
disagreeing, not the system failing.

Treat all three numbers as rough. There were only 11 emergencies in the messages your team had
labelled, and you can't learn much that's solid from 11 examples. Anyone who quotes you a confident
percentage off this data is overselling.

Two things I found that you should know. First, customers play down real problems — "not urgent, but
there's a smell of gas" is a real pattern, so gas, carbon monoxide, flooding and burning smells go
through a fixed rule that can only push urgency up, never down. No amount of polite phrasing gets
past it. Second, your own labels disagree with each other: near-identical messages were marked
differently on different days. That's just what triage looks like when you're busy, but it means
being *consistent* is a more realistic goal than being *right*.

## What I'd need to be confident

**Run it silently for a month.** It sorts in the background, your manager works exactly as she does
now, and we count what it would have buried. No customer is affected while it earns its place. This
is the most useful next step by a distance.

**Let her corrections feed it.** That's already built. Six months of her judgement is worth more
than anything clever I could do to the model.

**Give out-of-hours its own answer.** A quarter of your messages arrive when nobody's at the desk. A
gas message at 22:40 shouldn't sit in a sorted inbox — it should ring an on-call phone, and the
automatic reply should point the customer at the National Gas Emergency line, 0800 111 999. That's
your call, not a modelling problem.

## Two caveats

Message content would leave your systems. The bodies contain addresses and key-safe details, things
like "the neighbours have a spare key". The AI version sends all of that to an outside provider.
Decide that on purpose, with an agreement in place, rather than finding out later.

It only looks at one message at a time. It can't tell that this is the third time someone's chased
you, which is probably the most valuable thing it's still missing.

The attached `predictions.csv` is the AI system's own output on all 300 holdout messages, with the
reason it gave for each one. A simpler offline fallback exists for when the AI service is
unavailable; it shares the same safety rules but is markedly weaker on same-day jobs, so treat it as
degraded mode rather than an equal.
