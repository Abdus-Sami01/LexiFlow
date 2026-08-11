# LexiFlow

Real-time speech to structured insight, running entirely on your own CPU. Audio never leaves the
machine, nothing is written to disk mid-stream, and there is no API key anywhere in the codebase.

```
microphone ──► ring buffer ──► segmenter ──► whisper.cpp ──► analytics ──► state ──► dashboard
   thread 1                                    thread 2        thread 3            read-only
```

## What it does

- Captures raw microphone bytes straight into a pre-allocated RAM ring buffer, converts them to the
  16 kHz mono float32 layout Whisper requires, and cuts them into utterance-sized segments with an
  adaptive noise floor. No `.wav` files, no disk I/O in the hot path.
- Runs Whisper through native bindings (`pywhispercpp`, `whisper-cpp-python` or `faster-whisper`),
  passing the numpy array directly to C++ instead of shelling out to a binary.
- Extracts action items, deadlines, blockers, decisions, entities and sentiment from each line in
  well under a millisecond, using regex rules, a small spaCy model and a lexicon-based scorer.
- Keeps the microphone, inference and analytics on three isolated threads joined by bounded queues,
  so a slow transcription pass can never drop audio.
- Ships a Streamlit dashboard with a live transcript, action-item checklist, sentiment timeline and
  instant local search over the session.

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
```

As a library:

```python
from lexiflow import LexiFlowConfig, LexiFlowPipeline

config = LexiFlowConfig()
config.asr.model_path = "models/ggml-base.en.bin"

with LexiFlowPipeline(config) as pipeline:
    pipeline.subscribe(lambda event, item: print(item.text))
    input("recording, press enter to stop\n")

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
segment length, silence hangover, noise floor adaptation), `asr` (backend, model, threads, beam
size), `nlp` (which analyzers to enable) and `state` (database path, retention).

## Layout

| Path | Phase | Contents |
| --- | --- | --- |
| `lexiflow/audio/` | 1 | ring buffer, format conversion, segmenter, capture thread |
| `lexiflow/asr/` | 2 | hardware detection, native backends, transcription thread |
| `lexiflow/nlp/` | 3 | rule engine, entity extractor, sentiment, analytics pipeline |
| `lexiflow/state/` | 4 | thread-safe store, SQLite persistence and search, analytics thread |
| `lexiflow/ui/` | 5 | Streamlit dashboard |
| `lexiflow/pipeline.py` | — | orchestrator that owns the three threads and both queues |

## Tests

```bash
python -m pytest -q
python -m ruff check .
```

The suite covers the ring buffer, resampling, segmentation, every rule family, sentiment negation
and momentum, the store's persistence and search, and a full audio-to-insight pass through all
three threads using a scripted ASR backend.

## License

MIT
