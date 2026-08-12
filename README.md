# LexiFlow

[![ci](https://github.com/Abdus-Sami01/LexiFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdus-Sami01/LexiFlow/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![offline](https://img.shields.io/badge/cloud%20calls-zero-orange)](#)

Real-time speech to structured insight, running entirely on your own CPU. Audio never leaves the
machine, nothing is written to disk mid-stream, and there is no API key anywhere in the codebase.

![LexiFlow dashboard](docs/dashboard.png)

```
microphone ──► ring buffer ──► segmenter ──► whisper.cpp ──► analytics ──► state ──► dashboard
   thread 1                     │             thread 2        thread 3            read-only
                                └── partial hypotheses every 2s, dropped under backpressure
```

## What it does

- Captures raw microphone bytes straight into a pre-allocated RAM ring buffer, converts them to the
  16 kHz mono float32 layout Whisper requires, and cuts them into utterance-sized segments with an
  adaptive noise floor. No `.wav` files, no disk I/O in the hot path.
- Runs Whisper through native bindings (`pywhispercpp`, `whisper-cpp-python` or `faster-whisper`),
  passing the numpy array directly to C++ instead of shelling out to a binary.
- Extracts action items, deadlines, blockers, decisions, entities and sentiment from each line in
  well under a millisecond, using regex rules, a small spaCy model and a lexicon-based scorer.
  English, Spanish, French and German each have their own rule pack and valence lexicon; the
  language is detected per line and the analytics step aside entirely for anything else rather
  than emitting nonsense.
- Attributes each utterance to a speaker with online MFCC clustering: a numpy mel-filterbank and
  DCT frontend feeds cosine-similarity centroids that grow as new voices appear. When two people
  speak inside one segment it finds the change point and splits the segment there. Name a cluster
  once and the voiceprint is saved, so the same person keeps that name in later sessions.
- Summarises the session on demand with TextRank over a sentence-similarity graph, ranks keyphrases
  with RAKE, and flags topic changes by cosine distance between rolling word windows.
- Streams partial hypotheses while someone is still talking, so the transcript ticker updates
  mid-sentence instead of waiting for the pause. A governor drops them the moment the measured
  realtime factor says inference is falling behind, so they can never delay a final result.
- Gates the segmenter on spectral shape as well as loudness, so a fan, hiss or air conditioning
  no longer opens a segment the way a plain energy threshold would.
- Keeps the microphone, inference and analytics on three isolated threads joined by bounded queues,
  so a slow transcription pass can never drop audio.
- Ships a Streamlit dashboard with a live transcript, action-item checklist, sentiment timeline,
  speaker share bars, topic-shift log, an expandable digest, and search that spans every session
  ever recorded.

## Install

```bash
pip install -e ".[all]"          # everything
pip install -e ".[audio,ui]"     # capture + dashboard, bring your own ASR
python -m spacy download en_core_web_sm
```

Only `numpy` is mandatory. Every optional dependency degrades gracefully: without `sounddevice` you
can still replay files and inject text, without a Whisper backend the pipeline runs with the null
backend, and without spaCy or vaderSentiment the bundled regex and lexicon paths take over.

## Use

```bash
python -m lexiflow doctor              # hardware, backends, devices
python -m lexiflow models list         # catalogue of ggml weights and what is installed
python -m lexiflow models get base.en  # download it, resumable, into ~/.lexiflow/models
python -m lexiflow build               # tuned whisper.cpp build command for this machine
python -m lexiflow devices             # list input devices
python -m lexiflow run --model base.en
python -m lexiflow replay meeting.wav --model base.en   # drains the queue before exit
python -m lexiflow demo                # analytics over a sample conversation, no audio needed
python -m lexiflow dashboard           # Streamlit UI on :8501
python -m lexiflow bench               # analytics latency
python -m lexiflow sessions            # list everything recorded so far
python -m lexiflow search "budget"     # full-text search across every session
python -m lexiflow digest              # summarise the most recent session
python -m lexiflow export --format srt --format md --output notes
```

`--model` takes either a catalogue name (`base.en`) or a path to a `.bin`. Names resolve against
`~/.lexiflow/models`, overridable with `LEXIFLOW_MODELS`; a missing model tells you the exact
command to fetch it instead of failing deep inside the backend.

Exports cover `srt`, `vtt`, `txt`, `md` and `json`. Subtitle cues are made monotonic and
non-overlapping, and short utterances get a minimum on-screen duration, so the files load cleanly
in players that reject overlapping cues. The markdown export is a meeting-note document: summary,
keyphrases, action-item checkboxes, speaker table, entities and the full transcript. The same five
formats are one click away in the dashboard's session digest panel.

As a library:

```python
from lexiflow import LexiFlowConfig, LexiFlowPipeline

config = LexiFlowConfig()
config.asr.model_path = "models/ggml-base.en.bin"

with LexiFlowPipeline(config) as pipeline:
    pipeline.subscribe(lambda event, item: print(item.text))
    input("recording, press enter to stop\n")

print(pipeline.digest().as_markdown())
print(pipeline.store.speakers())
print(pipeline.store.export_json())
```

## Docker

```bash
docker build -t lexiflow .
docker run --rm -p 8501:8501 -v lexiflow-data:/data lexiflow
```

The image ships the dashboard on `0.0.0.0:8501` and keeps sessions and models in the `/data`
volume. Microphone capture inside a container needs the host's audio device passed through, which
is platform specific; file replay and text injection work out of the box.

## Building whisper.cpp for your CPU

Generic wheels are compiled for the lowest common denominator. `python -m lexiflow build` inspects
the host and prints the matching CMake invocation: AVX2/AVX512 plus OpenMP on Intel and AMD, the
Accelerate framework plus Metal on Apple Silicon.

```bash
git clone https://github.com/ggerganov/whisper.cpp
eval "$(python -m lexiflow build)"
bash whisper.cpp/models/download-ggml-model.sh base.en
```

## Configuration

Every knob lives in `lexiflow/config.py` and can be loaded from JSON:

```bash
python -m lexiflow --config my-settings.json run
```

The sections are `audio` (sample rates, block size, ring buffer length), `segmenter` (min/max
segment length, silence hangover, noise floor adaptation, spectral gate thresholds, partial
interval), `diarization` (similarity threshold, speaker cap, change-point splitting, saved
voiceprints), `asr` (backend, model, threads, beam size, word timestamps, realtime-factor
ceiling), `nlp` (which analyzers to enable, language detection, topic window, summary length) and
`state` (database path, retention).

Every advanced stage can be switched off independently. `segmenter.emit_partials = false` halves
CPU use on a slow machine, `segmenter.spectral_gate = false` returns to plain energy gating,
`diarization.enabled = false` skips the MFCC pass, `diarization.split_on_change = false` keeps
segments whole, and `nlp.detect_language = false` pins the analytics to one language. The rest of
the pipeline is unaffected.

## Layout

| Path | Phase | Contents |
| --- | --- | --- |
| `lexiflow/audio/` | 1 | ring buffer, format conversion, segmenter, capture thread, speaker tracker |
| `lexiflow/asr/` | 2 | hardware detection, native backends, transcription thread |
| `lexiflow/nlp/` | 3 | rules, entities, sentiment, summarisation, language routing, multilingual packs |
| `lexiflow/state/` | 4 | thread-safe store, SQLite persistence and search, analytics thread |
| `lexiflow/ui/` | 5 | Streamlit dashboard |
| `lexiflow/export.py` | — | srt, vtt, txt, markdown and json writers |
| `lexiflow/pipeline.py` | — | orchestrator that owns the three threads and both queues |

## Tests

```bash
python -m pytest -q
python -m ruff check .
```

125 tests cover the ring buffer, resampling, segmentation, the spectral gate against real noise
and rumble, partial emission and the realtime-factor governor, every rule family in four
languages, language detection and its refusal to guess, sentiment negation and momentum, the MFCC
frontend, speaker clustering, change-point splitting and voiceprint round-trips, TextRank, RAKE,
topic drift, filler compression, subtitle cue timing from backend spans, every export format, the
model catalogue, the store's persistence and cross-session search, queue draining against a
deliberately slow backend, and full audio-to-insight passes through all three threads.

## Limitations

What is still true, stated plainly:

- **Analytics covers four languages, not every language.** English, Spanish, French and German
  have rule packs and lexicons. Detection recognises those plus Italian, Portuguese and Dutch, and
  for anything outside the supported four the rules and sentiment switch themselves off — you get
  a clean transcript and no invented insight. Adding a language means adding a lexicon and a rule
  pack to `lexiflow/nlp/multilingual.py`; nothing else changes.
- **The sentiment lexicons for es/fr/de are compact.** Roughly fifty hand-picked terms each,
  against a few thousand for English via vaderSentiment. They get the polarity right on clear
  statements and will miss subtler wording.
- **Speaker attribution is still unsupervised.** Distinct voices separate cleanly and mid-segment
  changes are now split, but genuinely simultaneous speech is one waveform and cannot be
  un-mixed by clustering. Very similar voices can merge into one cluster. Labels are per segment,
  not per word.
- **The spectral gate rejects noise, not music.** Fans, hiss and broadband noise are filtered by
  flatness and zero-crossing rate. Tonal music sits in the same part of the feature space as
  voiced speech and will still open a segment.
- **Summarisation is extractive.** TextRank picks the most central sentences actually spoken and
  strips verbal filler from them; it never writes a sentence nobody said. Genuine abstraction
  needs a language model, which would break the zero-cloud promise this project is built on.
- **Latency still depends on the model.** That is arithmetic, not a bug: `base.en` runs faster
  than realtime on a modern laptop, `large-v3` does not. What is handled is the consequence —
  the pipeline measures its own realtime factor, sheds partials when it slips, and reports
  `keeping_up: false` instead of silently building a backlog.

## Roadmap

- A `textual` terminal dashboard alongside the Streamlit one
- Per-word subtitle cues, now that word timings are already carried through
- More language packs, starting with Italian and Portuguese

## License

MIT
