"""Offline translation: Argos/OPUS-MT for text, Whisper's own task for speech."""

from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..config import TranslationConfig
from ..observability import record_failure

_REGISTRY: Dict[str, type] = {}


class TranslationUnavailable(RuntimeError):
    """Raised when no local engine can serve the requested language pair."""


@dataclass
class TranslationResult:
    text: str
    source: str
    target: str
    engine: str = "identity"
    elapsed_ms: float = 0.0
    cached: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "source": self.source,
            "target": self.target,
            "engine": self.engine,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "cached": self.cached,
        }


def register(name: str) -> Callable[[type], type]:
    def decorator(klass: type) -> type:
        klass.name = name
        _REGISTRY[name] = klass
        return klass

    return decorator


class Translator:
    """Every engine takes plain text and returns plain text, nothing else."""

    name = "base"
    priority = 100

    def __init__(self, config: Optional[TranslationConfig] = None) -> None:
        self.config = config or TranslationConfig()

    @classmethod
    def is_available(cls) -> bool:
        raise NotImplementedError

    def supports(self, source: str, target: str) -> bool:
        raise NotImplementedError

    def translate(self, text: str, source: str, target: str) -> str:
        raise NotImplementedError

    def installed_pairs(self) -> List[Tuple[str, str]]:
        return []


@register("identity")
class IdentityTranslator(Translator):
    """The honest no-op: used when source and target already match."""

    priority = 900

    @classmethod
    def is_available(cls) -> bool:
        return True

    def supports(self, source: str, target: str) -> bool:
        return source == target

    def translate(self, text: str, source: str, target: str) -> str:
        return text


@register("argos")
class ArgosTranslator(Translator):
    """argostranslate wraps CTranslate2 OPUS-MT models, all running locally."""

    priority = 10

    def __init__(self, config: Optional[TranslationConfig] = None) -> None:
        super().__init__(config)
        self._module = None
        self._package_module = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            importlib.import_module("argostranslate.translate")
            return True
        except Exception:
            return False

    def _translate_module(self):
        if self._module is None:
            self._module = importlib.import_module("argostranslate.translate")
        return self._module

    def _packages(self):
        if self._package_module is None:
            self._package_module = importlib.import_module("argostranslate.package")
        return self._package_module

    def installed_pairs(self) -> List[Tuple[str, str]]:
        try:
            languages = self._translate_module().get_installed_languages()
        except Exception:
            return []
        pairs = []
        for source in languages:
            for target in languages:
                if source.code == target.code:
                    continue
                try:
                    if source.get_translation(target) is not None:
                        pairs.append((source.code, target.code))
                except Exception:
                    continue
        return pairs

    def supports(self, source: str, target: str) -> bool:
        if source == target:
            return True
        return (source, target) in set(self.installed_pairs())

    def translate(self, text: str, source: str, target: str) -> str:
        if source == target:
            return text
        module = self._translate_module()
        return str(module.translate(text, source, target))

    def install_pair(self, source: str, target: str) -> bool:
        """Download and install one language pair; needs network exactly once."""
        packages = self._packages()
        packages.update_package_index()
        for candidate in packages.get_available_packages():
            if candidate.from_code == source and candidate.to_code == target:
                packages.install_from_path(candidate.download())
                return True
        return False


class TranslationEngine:
    """Picks an engine per pair, memoises repeated lines, never blocks on failure."""

    def __init__(
        self, config: Optional[TranslationConfig] = None, translator: Optional[Translator] = None
    ) -> None:
        self.config = config or TranslationConfig()
        self.translator = translator or create_translator(self.config)
        self.identity = IdentityTranslator(self.config)
        self.failures = 0
        self.translated = 0
        self.total_ms = 0.0
        self._cache: Dict[Tuple[str, str, str], str] = {}
        self._lock = threading.Lock()

    @property
    def engine_name(self) -> str:
        return self.translator.name

    @property
    def average_ms(self) -> float:
        return self.total_ms / self.translated if self.translated else 0.0

    def supports(self, source: str, target: str) -> bool:
        return source == target or self.translator.supports(source, target)

    def translate(
        self, text: str, source: str, target: Optional[str] = None
    ) -> Optional[TranslationResult]:
        """Return None when there is nothing to do or nothing that can do it."""
        destination = target or self.config.target_language
        cleaned = (text or "").strip()
        if not cleaned or not destination:
            return None
        if source == destination:
            return None

        key = (cleaned, source, destination)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return TranslationResult(hit, source, destination, self.engine_name, 0.0, True)

        if not self.translator.supports(source, destination):
            self.failures += 1
            return None

        started = time.perf_counter()
        try:
            rendered = self.translator.translate(cleaned, source, destination)
        except Exception as error:
            record_failure("translate", error, pair=f"{source}->{destination}")
            self.failures += 1
            return None
        elapsed = (time.perf_counter() - started) * 1000.0

        with self._lock:
            if len(self._cache) >= self.config.cache_size:
                self._cache.clear()
            self._cache[key] = rendered
            self.translated += 1
            self.total_ms += elapsed

        return TranslationResult(rendered, source, destination, self.engine_name, elapsed)

    def stats(self) -> Dict[str, object]:
        return {
            "engine": self.engine_name,
            "translated": self.translated,
            "failures": self.failures,
            "average_ms": round(self.average_ms, 3),
            "pairs": [f"{a}->{b}" for a, b in self.translator.installed_pairs()],
        }


def available_translators() -> List[str]:
    return [
        name
        for name, klass in sorted(_REGISTRY.items(), key=lambda item: item[1].priority)
        if klass.is_available()
    ]


def create_translator(config: Optional[TranslationConfig] = None) -> Translator:
    config = config or TranslationConfig()
    if config.backend and config.backend != "auto":
        klass = _REGISTRY.get(config.backend)
        if klass is None:
            raise TranslationUnavailable(f"unknown translation backend '{config.backend}'")
        return klass(config)

    for _, klass in sorted(_REGISTRY.items(), key=lambda item: item[1].priority):
        if klass.is_available():
            return klass(config)
    return IdentityTranslator(config)


@dataclass
class TranslationReport:
    backend: str
    available: List[str] = field(default_factory=list)
    pairs: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {"backend": self.backend, "available": self.available, "pairs": self.pairs}


def report(config: Optional[TranslationConfig] = None) -> TranslationReport:
    translator = create_translator(config)
    return TranslationReport(
        backend=translator.name,
        available=available_translators(),
        pairs=[f"{a}->{b}" for a, b in translator.installed_pairs()],
    )
