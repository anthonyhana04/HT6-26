# AI Council

A real-time, event-driven **multi-agent orchestration engine**. Four AI
personalities — a Lead, a Thinker, a Second Guesser and a Brutalist — sit around
a table and decide, on their own, whether they have anything worth saying. Not
everyone talks. Simple questions get one answer; open-ended ones spark a real
discussion.

This is **not** a chatbot and **not** a fan-out wrapper over several LLM APIs.
The product is the orchestration: proposal-based turn-taking, a pure-Python
moderator, a speaking queue with interrupts, and a strict event bus that lets
future hardware (voice, LEDs, mics, vision) plug in without touching business
logic.

The first milestone is a CLI. The architecture already anticipates the rest.

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # optional: add provider API keys

python -m app.main          # interactive council
```

No API keys? It just works. Any member without a key (or when
`AI_COUNCIL_FORCE_MOCK=true`) falls back to a deterministic offline "brain", so
the **orchestration engine is fully demonstrable with zero setup**.

Try it:

- `what is 2+2?` → only the **Lead** answers; everyone else stays silent.
- `I'm considering quitting school to start a company.` → several members engage,
  someone raises an **interrupt**, and the Lead closes with a summary.

A non-interactive scripted demo (great for screenshots/CI):

```bash
AI_COUNCIL_FORCE_MOCK=true python scripts/demo.py
```

CLI commands: `/help`, `/verbose` (toggle the moderator's reasoning log),
`/members`, `/quit`.

---

## The two-phase turn

Every user message runs through two phases. This is what keeps token usage low
and conversations natural.

```
 UserMessage
     │
     ▼
 ┌──────────────────────── PHASE 1 · PROPOSAL ────────────────────────┐
 │ Every member gets the full context and returns ONLY a tiny bid:    │
 │   { should_speak, confidence, intent, reason, target? }            │
 │ No prose is generated. Silence is a valid, common answer.          │
 └────────────────────────────────────────────────────────────────────┘
     │
     ▼
 ┌──────────────────── MODERATOR (pure Python) ───────────────────────┐
 │ filter (should_speak + confidence) → rank → collapse dominant      │
 │ answers to a single speaker → cap → order (Lead opens, corrections │
 │ follow their target) → enforce per-turn fairness                   │
 └────────────────────────────────────────────────────────────────────┘
     │
     ▼
 ┌──────────────────── PHASE 2 · GENERATION ──────────────────────────┐
 │ ONLY the selected members generate a full response.                │
 │ Each is spoken via a SpeakEvent; the next speaker waits for the     │
 │ current SpeechFinished (no overlap).                                │
 │ After each turn, an interrupt round lets others react.             │
 └────────────────────────────────────────────────────────────────────┘
```

Possible **intents**: `ANSWER`, `QUESTION`, `CORRECTION`, `DISAGREEMENT`,
`AGREEMENT`, `FOLLOW_UP`, `SUMMARY`, `OBSERVATION`.

---

## Architecture

Strictly layered. Dependencies point downward; nothing in the core imports a
peripheral. The **event bus** is the only integration seam.

```
app/
  models/      Immutable value objects (Message, Proposal, Response, AgentContext)
  events/      Event vocabulary + the async EventBus (pub/sub, wait_for)
  memory/      Conversation log + History store (multi-conversation ready)
  agents/      BaseAgent two-phase brain + provider adapters + offline Mock
  council/     Pure-Python core: arbitration, moderator, queue, scheduler, Council
  speech/      SpeakEvent → SpeechBackend (terminal today, ElevenLabs tomorrow)
  config/      Settings, agent profiles, and the DI factory (composition root)
  main.py      CLI — a *subscriber*, not a driver
```

### Responsibilities

- **Moderator** (`council/moderator.py`) — the chairperson, and explicitly *not*
  an LLM. Requests proposals, ranks them, enforces fairness and the speaker cap,
  decides which interrupts are honoured. Every decision is emitted as an event.
- **Arbitrator** (`council/arbitration.py`) — pure, deterministic ranking policy
  (threshold, dominant-answer collapse, cap, ordering). Trivially unit-testable.
- **Scheduler** (`council/scheduler.py`) — drives one user message end to end:
  proposal round → selection → queue → generate/speak per speaker → interrupt
  rounds → optional Lead summary.
- **SpeakingQueue** (`council/queue.py`) — FIFO for fairness, with a privileged
  front for accepted interrupts.
- **EventBus** (`events/event_bus.py`) — async pub/sub. Handlers may be sync or
  async, run concurrently, and are isolated (one failure never breaks a
  publish). `wait_for` turns it into a coordination primitive.

### Events (the contract with all peripherals)

`UserMessage`, `ProposalCreated`, `ProposalAccepted`, `ProposalRejected`,
`SpeechQueued`, `InterruptRequest`, `SpeakEvent`, `SpeechStarted`,
`SpeechProgress`, `SpeechFinished`, `ConversationEnded`.

---

## Interrupts

Agents **never** cut off live speech. While someone speaks, others may raise an
`InterruptRequest`. The moderator (using a higher confidence bar) decides whether
to slot the requester to the **front** of the queue — so they speak *next*, once
the current speaker finishes. No overlapping speakers, ever.

---

## Speech, voices and LEDs

The rest of the app **never** calls a TTS API. The council only emits
`SpeakEvent(speaker, text, intent)`. A `SpeechBackend` renders it and the
`SpeechService` translates the render into lifecycle events:

```
SpeakEvent → SpeechStarted → SpeechProgress(amplitude)* → SpeechFinished
```

- **Terminal:** `speech/player.py` prints to the terminal and synthesizes an
  amplitude envelope, so the event stream behaves identically with or without
  audio.
- **Voice (ElevenLabs):** `speech/elevenlabs.py` streams TTS audio chunks,
  plays them through the speaker in real time (`sounddevice`), and carries the
  **real** per-chunk amplitude on `SpeechProgress`. Flow becomes
  `SpeakEvent → ElevenLabs → PCM stream → speaker`. **Each member gets a unique
  voice**, and **voice IDs live in configuration**, never in code.
- Selection is config-driven (`AI_COUNCIL_SPEECH` = `auto` | `elevenlabs` |
  `terminal`); `config/factory._build_speech_backend` is the only switch. Set
  `ELEVENLABS_API_KEY` and voices turn on automatically.
- **Unique voices:** set `DEEPSEEK_VOICE_ID` / `ANTHROPIC_VOICE_ID` /
  `GEMINI_VOICE_ID` / `GROQ_VOICE_ID`, or leave any blank and the backend
  auto-assigns a distinct voice from your account at startup.
- **LEDs (future):** the Raspberry Pi lamp subscribes to `SpeechStarted` /
  `SpeechProgress` / `SpeechFinished` — now driven by real audio loudness.
  Nothing outside the speech layer knows LEDs exist. Adding hardware = adding a
  subscriber, not editing logic.

---

## Configuration

All tunables live in `.env` (see `.env.example`) and are loaded via
`app/config/settings.py`:

| Variable | Meaning |
| --- | --- |
| `AI_COUNCIL_FORCE_MOCK` | Force every member to the offline brain |
| `AI_COUNCIL_MIN_CONFIDENCE` | Confidence needed to be scheduled |
| `AI_COUNCIL_INTERRUPT_CONFIDENCE` | Confidence needed to interrupt |
| `AI_COUNCIL_MAX_SPEAKERS` | Max speakers per proposal round |
| `AI_COUNCIL_MAX_TURNS` | Hard cap on turns per user message |
| `AI_COUNCIL_MAX_TURNS_PER_AGENT` | Per-member fairness cap |
| `DEEPSEEK_API_KEY` … `GROQ_API_KEY` | Provider keys (missing → mock) |
| `*_MODEL` | Per-provider model overrides |
| `AI_COUNCIL_SPEECH` | `auto` \| `elevenlabs` \| `terminal` |
| `ELEVENLABS_API_KEY` | Turns on real voices |
| `ELEVENLABS_MODEL` | TTS model (default `eleven_flash_v2_5`) |
| `*_VOICE_ID` | Per-member voice (blank → auto-assigned) |

---

## Extending the council

- **New member / provider:** subclass `BaseAgent` and implement one method,
  `_complete(system, user, *, json_mode, ...)`. Add a profile in
  `config/profiles.py` and wire it in `config/factory.py`.
- **New behaviour policy:** the moderator and arbitrator are pure Python — tune
  thresholds or ordering without touching agents or I/O.
- **New peripheral (audio, LEDs, logging, metrics, a web UI):** subscribe to the
  relevant events on the bus. The core never needs to know you exist.

## Requirements

Python 3.12+ (developed/tested on 3.13), asyncio, Pydantic, Rich. Provider SDKs
(`openai` — also used for DeepSeek's OpenAI-compatible API — `anthropic`,
`google-genai`, `groq`) are optional at runtime and
imported lazily — a missing SDK simply routes that member to the mock brain.
