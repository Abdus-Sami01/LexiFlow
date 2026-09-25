"""Build a structurally real CTranslate2 Whisper model without downloading anything.

The weights are random, so the text it produces is meaningless. Everything around
them is real: a real model directory that real faster-whisper loads and runs. That
is the one thing a scripted backend can never check, and the weights are the only
part of a Whisper install that cannot be reconstructed offline.
"""

import json
import sys
from pathlib import Path

import numpy as np
from ctranslate2.specs.whisper_spec import WhisperSpec

VOCAB = 51865
MELS = 80
CTX = 448
FRAMES = 1500


def fill(spec, dim, heads, layers, rng):
    def randn(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    def ones(*shape):
        return np.ones(shape, dtype=np.float32)

    def zeros(*shape):
        return np.zeros(shape, dtype=np.float32)

    def norm(target):
        target.gamma = ones(dim)
        target.beta = zeros(dim)

    def attention(target, self_attention):
        norm(target.layer_norm)
        fan = 3 if self_attention else 1
        target.linear[0].weight = randn(dim * fan, dim)
        target.linear[0].bias = zeros(dim * fan)
        if not self_attention:
            target.linear[1].weight = randn(dim * 2, dim)
            target.linear[1].bias = zeros(dim * 2)
            target.linear[2].weight = randn(dim, dim)
            target.linear[2].bias = zeros(dim)
        else:
            target.linear[1].weight = randn(dim, dim)
            target.linear[1].bias = zeros(dim)

    def feed_forward(target):
        norm(target.layer_norm)
        target.linear_0.weight = randn(dim * 4, dim)
        target.linear_0.bias = zeros(dim * 4)
        target.linear_1.weight = randn(dim, dim * 4)
        target.linear_1.bias = zeros(dim)

    encoder = spec.encoder
    encoder.conv1.weight = randn(dim, MELS, 3)
    encoder.conv1.bias = zeros(dim)
    encoder.conv2.weight = randn(dim, dim, 3)
    encoder.conv2.bias = zeros(dim)
    encoder.position_encodings.encodings = randn(FRAMES, dim)
    norm(encoder.layer_norm)
    for layer in encoder.layer:
        attention(layer.self_attention, True)
        feed_forward(layer.ffn)

    decoder = spec.decoder
    decoder.embeddings.weight = randn(VOCAB, dim)
    decoder.position_encodings.encodings = randn(CTX, dim)
    norm(decoder.layer_norm)
    decoder.projection.weight = decoder.embeddings.weight
    for layer in decoder.layer:
        attention(layer.self_attention, True)
        attention(layer.attention, False)
        feed_forward(layer.ffn)
    return spec


def base_alphabet() -> list:
    """The byte-level alphabet the tokenizer's pre-tokenizer actually emits."""
    from tokenizers.pre_tokenizers import ByteLevel

    return sorted(ByteLevel.alphabet())


def token_list() -> list:
    """The same vocabulary the tokenizer file carries, in id order."""
    tokens = list(base_alphabet())
    seen = set(tokens)
    for first in tokens[:]:
        for second in tokens[:]:
            pair = first + second
            if pair not in seen and len(tokens) < VOCAB - 1600:
                tokens.append(pair)
                seen.add(pair)
    specials = special_tokens()
    room = VOCAB - len(specials)
    while len(tokens) < room:
        tokens.append(f"tok{len(tokens)}")
    return tokens[:room] + specials


def special_tokens() -> list:
    """The reserved ids Whisper reserves at the top of its vocabulary."""
    specials = ["<|endoftext|>", "<|startoftranscript|>"]
    specials += [f"<|{code}|>" for code in LANGUAGES]
    specials += ["<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>",
                 "<|nospeech|>", "<|notimestamps|>"]
    specials += [f"<|{index * 0.02:.2f}|>" for index in range(1501)]
    return specials


def tokenizer_json(path: Path) -> None:
    """A byte-level vocabulary the size Whisper expects, with the special tokens named."""
    tokens = token_list()
    vocab = {token: index for index, token in enumerate(tokens)}
    specials = special_tokens()
    start = VOCAB - len(specials)
    added = []
    for offset, token in enumerate(specials):
        identifier = start + offset
        vocab[token] = identifier
        added.append(
            {
                "id": identifier,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )

    payload = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                          "trim_offsets": True, "use_regex": True},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": []},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


LANGUAGES = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv",
    "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr",
    "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
    "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
]


def build(target: Path, dim: int = 64, heads: int = 2, layers: int = 2, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    spec = WhisperSpec(layers, heads, layers, heads)
    fill(spec, dim, heads, layers, rng)
    spec.config.suppress_ids = [VOCAB - 1]
    spec.config.suppress_ids_begin = [VOCAB - 1]
    first_language = VOCAB - len(LANGUAGES) - 1509
    spec.config.lang_ids = [first_language + index for index in range(len(LANGUAGES))]
    spec.config.alignment_heads = [[0, 0]]
    spec.register_vocabulary(token_list())
    spec.validate()
    target.mkdir(parents=True, exist_ok=True)
    spec.save(str(target))
    tokenizer_json(target / "tokenizer.json")
    (target / "preprocessor_config.json").write_text(
        json.dumps({"feature_size": MELS, "sampling_rate": 16000, "chunk_length": 30}),
        encoding="utf-8",
    )
    return target


if __name__ == "__main__":
    where = Path(sys.argv[1] if len(sys.argv) > 1 else "ct2-random-whisper")
    print(build(where))
