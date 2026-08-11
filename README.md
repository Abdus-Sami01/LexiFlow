# LexiFlow

Real-time speech to structured insight, running entirely on your own CPU. Audio never leaves the
machine, nothing is written to disk mid-stream, and there is no API key anywhere in the codebase.

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
- Attributes each utterance to a speaker with online MFCC clustering: a numpy mel-filterbank and
  DCT frontend feeds cosine-similarity centroids that grow as new voices appear. No pretrained
  embedding model, no enrolment step.
- Summarises the session on demand with TextRank over a sentence-similarity graph, ranks keyphrases
  with RAKE, and flags topic changes by cosine distance between rolling word windows.
- Streams partial hypotheses while someone is still talking, so the transcript ticker updates
  mid-sentence instead of waiting for the pause. Partials are dropped automatically whenever the
  inference queue is busy, so they can never delay a final result.
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
python -m lexiflow build               # tuned whisper.cpp build command for this machine
python -m lexiflow devices             # list input devices
python -m lexiflow run --model models/ggml-base.en.bin
python -m lexiflow replay meeting.wav --model models/ggml-base.en.bin
python -m lexiflow demo                # analytics over a sample conversation, no audio needed
python -m lexiflow dashboard           # Streamlit UI on :8501
python -m lexiflow bench               # analytics latency
python -m lexiflow sessions            # list everything recorded so far
python -m lexiflow search "budget"     # full-text search across every session
python -m lexiflow digest              # summarise the most recent session
```

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
segment length, silence hangover, noise floor adaptation, partial interval), `diarization`
(similarity threshold, speaker cap, adaptation rate), `asr` (backend, model, threads, beam size),
`nlp` (which analyzers to enable, topic window, summary length) and `state` (database path,
retention).

Every advanced stage can be switched off independently. `segmenter.emit_partials = false` halves
CPU use on a slow machine, `diarization.enabled = false` skips the MFCC pass, and
`nlp.enable_topics = false` drops topic tracking. The rest of the pipeline is unaffected.

## Layout

| Path | Phase | Contents |
| --- | --- | --- |
| `lexiflow/audio/` | 1 | ring buffer, format conversion, segmenter, capture thread, speaker tracker |
| `lexiflow/asr/` | 2 | hardware detection, native backends, transcription thread |
| `lexiflow/nlp/` | 3 | rule engine, entity extractor, sentiment, summarisation, analytics pipeline |
| `lexiflow/state/` | 4 | thread-safe store, SQLite persistence and search, analytics thread |
| `lexiflow/ui/` | 5 | Streamlit dashboard |
| `lexiflow/pipeline.py` | — | orchestrator that owns the three threads and both queues |

## Tests

```bash
python -m pytest -q
python -m ruff check .
```

63 tests cover the ring buffer, resampling, segmentation and partial emission, every rule family,
sentiment negation and momentum, the MFCC frontend and speaker clustering, TextRank, RAKE and topic
drift, the store's persistence and cross-session search, and two full audio-to-insight passes
through all three threads using a scripted ASR backend.

## License

MIT
