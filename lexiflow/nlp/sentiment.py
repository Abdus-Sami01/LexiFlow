"""Lexicon and arithmetic sentiment: no neural network, no cloud, no latency."""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional

TOKEN_PATTERN = re.compile(r"[A-Za-z']+|[!?]+")

NEGATIONS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere",
    "cannot", "cant", "can't", "wont", "won't", "dont", "don't", "isnt", "isn't",
    "arent", "aren't", "wasnt", "wasn't", "didnt", "didn't", "doesnt", "doesn't",
    "shouldnt", "shouldn't", "wouldnt", "wouldn't", "couldnt", "couldn't", "without",
    "hardly", "barely", "rarely", "scarcely", "lack", "lacks", "lacking",
}

BOOSTERS: Dict[str, float] = {
    "absolutely": 0.293, "amazingly": 0.293, "completely": 0.293, "considerably": 0.193,
    "decidedly": 0.293, "deeply": 0.293, "enormously": 0.293, "entirely": 0.293,
    "especially": 0.293, "exceptionally": 0.293, "extremely": 0.293, "fabulously": 0.293,
    "fully": 0.293, "greatly": 0.293, "highly": 0.293, "hugely": 0.293, "incredibly": 0.293,
    "intensely": 0.293, "majorly": 0.293, "more": 0.193, "most": 0.293, "particularly": 0.193,
    "purely": 0.293, "quite": 0.193, "really": 0.293, "remarkably": 0.293, "so": 0.193,
    "substantially": 0.293, "thoroughly": 0.293, "totally": 0.293, "tremendously": 0.293,
    "unbelievably": 0.293, "utterly": 0.293, "very": 0.293, "super": 0.293,
    "almost": -0.293, "barely": -0.293, "hardly": -0.293, "kinda": -0.293, "kind": -0.150,
    "less": -0.293, "little": -0.293, "marginally": -0.293, "occasionally": -0.293,
    "partly": -0.293, "scarcely": -0.293, "slightly": -0.293, "somewhat": -0.293,
}

BASE_LEXICON: Dict[str, float] = {
    "abandon": -1.9, "able": 1.2, "abuse": -2.7, "accept": 1.3, "accomplished": 2.4,
    "accurate": 1.6, "advantage": 1.7, "afraid": -1.9, "agree": 1.5, "alarming": -2.1,
    "amazing": 3.0, "angry": -2.5, "annoy": -1.8, "annoyed": -2.0, "anxious": -1.8,
    "appreciate": 2.2, "approve": 1.8, "asset": 1.4, "attack": -2.4, "awesome": 3.1,
    "awful": -2.7, "awkward": -1.4, "bad": -2.5, "badly": -2.2, "beautiful": 2.8,
    "best": 3.2, "better": 1.9, "blame": -2.0, "block": -1.3, "blocked": -1.8,
    "blocker": -2.0, "boring": -1.7, "brilliant": 2.9, "broken": -2.3, "bug": -1.5,
    "calm": 1.3, "careless": -2.0, "champion": 2.4, "chaos": -2.4, "clean": 1.6,
    "clear": 1.4, "clever": 2.1, "comfortable": 1.9, "complain": -1.9, "concern": -1.4,
    "concerned": -1.6, "confident": 2.2, "confused": -1.6, "confusing": -1.8,
    "congrats": 2.8, "congratulations": 3.0, "cool": 1.9, "correct": 1.6, "crash": -2.4,
    "crashed": -2.5, "crazy": -1.1, "critical": -1.7, "damage": -2.4, "danger": -2.5,
    "dead": -2.3, "delay": -1.8, "delayed": -1.9, "delight": 2.7, "deny": -1.6,
    "difficult": -1.6, "disappointed": -2.3, "disaster": -3.0, "dislike": -1.9,
    "doubt": -1.4, "dread": -2.2, "easy": 1.8, "efficient": 2.0, "elegant": 2.2,
    "embarrassed": -1.8, "encourage": 2.0, "enjoy": 2.3, "excellent": 2.9, "excited": 2.6,
    "exciting": 2.5, "expensive": -1.3, "fail": -2.5, "failed": -2.6, "failure": -2.7,
    "fantastic": 3.0, "fault": -2.0, "favorite": 2.4, "fear": -2.4, "fine": 1.2,
    "fix": 1.3, "fixed": 1.8, "flawless": 2.8, "fortunate": 2.2, "frustrated": -2.3,
    "frustrating": -2.4, "fun": 2.3, "glad": 2.2, "good": 1.9, "grateful": 2.6,
    "great": 2.7, "happy": 2.7, "hard": -1.1, "hate": -3.0, "healthy": 2.0, "help": 1.7,
    "helpful": 2.2, "hopeless": -2.6, "horrible": -2.9, "hurt": -2.2, "ideal": 2.3,
    "impossible": -2.2, "impressed": 2.4, "impressive": 2.5, "improve": 1.8,
    "improvement": 1.9, "incredible": 2.8, "issue": -1.4, "lag": -1.6, "late": -1.5,
    "learn": 1.3, "like": 1.5, "love": 3.0, "lovely": 2.6, "lucky": 2.1, "mess": -2.1,
    "messy": -1.9, "miss": -1.3, "mistake": -2.0, "nasty": -2.6, "neat": 1.9,
    "nervous": -1.7, "nice": 1.9, "outstanding": 3.0, "overwhelmed": -2.1, "pain": -2.3,
    "painful": -2.4, "panic": -2.6, "perfect": 3.0, "pleasant": 2.2, "please": 1.3,
    "pleased": 2.3, "positive": 2.1, "powerful": 2.2, "praise": 2.5, "problem": -1.9,
    "productive": 2.1, "progress": 1.9, "promising": 2.1, "proud": 2.5, "quick": 1.4,
    "reliable": 2.2, "relieved": 2.0, "reject": -2.0, "resolve": 1.8, "resolved": 2.0,
    "risk": -1.6, "risky": -1.9, "robust": 2.1, "sad": -2.4, "safe": 1.9, "satisfied": 2.3,
    "scared": -2.3, "secure": 1.9, "serious": -1.2, "shame": -2.2, "sharp": 1.6,
    "slow": -1.6, "smart": 2.2, "smooth": 2.0, "solid": 2.0, "solve": 1.9, "solved": 2.1,
    "sorry": -1.3, "stable": 1.9, "stress": -2.2, "stressed": -2.3, "stressful": -2.3,
    "stuck": -2.0, "stupid": -2.6, "success": 2.7, "successful": 2.7, "superb": 2.9,
    "support": 1.8, "sure": 1.3, "terrible": -2.8, "terrific": 2.8, "thank": 2.2,
    "thanks": 2.2, "threat": -2.4, "tired": -1.6, "tough": -1.4, "trouble": -2.1,
    "trust": 2.2, "ugly": -2.4, "unacceptable": -2.7, "uncertain": -1.5, "unhappy": -2.4,
    "unstable": -2.1, "upset": -2.3, "urgent": -1.4, "useful": 2.0, "useless": -2.4,
    "victory": 2.7, "waste": -2.2, "weak": -1.9, "welcome": 2.0, "well": 1.4,
    "win": 2.6, "wonderful": 2.9, "worried": -2.1, "worry": -2.0, "worse": -2.4,
    "worst": -3.0, "wrong": -2.2,
}

ALPHA = 15.0
CAPS_BOOST = 0.733
EXCLAMATION_BOOST = 0.292
QUESTION_BOOST = 0.18
CONTRAST_DAMPING = 0.5
NEGATION_SCALE = -0.74


@dataclass
class SentimentScore:
    compound: float
    positive: float
    negative: float
    neutral: float
    label: str
    engine: str = "lexiflow"

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "compound": self.compound,
            "positive": self.positive,
            "negative": self.negative,
            "neutral": self.neutral,
            "label": self.label,
            "engine": self.engine,
        }


def label_for(compound: float, threshold: float = 0.05) -> str:
    if compound >= threshold:
        return "positive"
    if compound <= -threshold:
        return "negative"
    return "neutral"


def _normalise(total: float) -> float:
    return float(max(-1.0, min(1.0, total / math.sqrt(total * total + ALPHA))))


class LexiconSentimentAnalyzer:
    """Pure arithmetic VADER-style scorer over the bundled lexicon."""

    engine = "lexiflow-lexicon"

    def __init__(
        self, lexicon: Optional[Dict[str, float]] = None, language: str = "en"
    ) -> None:
        self.language = language
        self.lexicon = dict(BASE_LEXICON) if language == "en" else {}
        self.negations = set(NEGATIONS)
        self.boosters = dict(BOOSTERS)

        if language != "en":
            from .multilingual import BOOSTERS as EXTRA_BOOSTERS
            from .multilingual import NEGATIONS as EXTRA_NEGATIONS
            from .multilingual import lexicon_for

            self.lexicon.update(lexicon_for(language))
            self.negations = set(EXTRA_NEGATIONS.get(language, NEGATIONS))
            self.boosters = dict(EXTRA_BOOSTERS.get(language, BOOSTERS))

        if lexicon:
            self.lexicon.update(lexicon)

    def polarity_scores(self, text: str) -> SentimentScore:
        raw_tokens = TOKEN_PATTERN.findall(text or "")
        if not raw_tokens:
            return SentimentScore(0.0, 0.0, 0.0, 1.0, "neutral", self.engine)

        lowered = [token.lower() for token in raw_tokens]
        all_caps_words = [
            token for token in raw_tokens if token.isupper() and len(token) > 1 and token.isalpha()
        ]
        shouting = 0 < len(all_caps_words) < len([t for t in raw_tokens if t.isalpha()])

        valences: List[float] = []
        for position, word in enumerate(lowered):
            valence = self.lexicon.get(word)
            if valence is None:
                continue
            if shouting and raw_tokens[position].isupper():
                valence += CAPS_BOOST * (1 if valence > 0 else -1)
            valence += self._booster_delta(lowered, position, valence)
            valence *= self._negation_factor(lowered, position)
            valences.append(valence)

        if not valences:
            return SentimentScore(0.0, 0.0, 0.0, 1.0, "neutral", self.engine)

        valences = self._apply_contrast(lowered, valences)
        total = sum(valences) + self._punctuation_emphasis(text, sum(valences))
        compound = _normalise(total)

        positive_sum = sum(value + 1.0 for value in valences if value > 0)
        negative_sum = sum(value - 1.0 for value in valences if value < 0)
        neutral_count = len(lowered) - len(valences)
        magnitude = positive_sum + abs(negative_sum) + neutral_count
        if magnitude <= 0:
            return SentimentScore(compound, 0.0, 0.0, 1.0, label_for(compound), self.engine)

        return SentimentScore(
            compound=round(compound, 4),
            positive=round(positive_sum / magnitude, 4),
            negative=round(abs(negative_sum) / magnitude, 4),
            neutral=round(neutral_count / magnitude, 4),
            label=label_for(compound),
            engine=self.engine,
        )

    def _booster_delta(self, tokens: List[str], position: int, valence: float) -> float:
        delta = 0.0
        for distance in range(1, 4):
            index = position - distance
            if index < 0:
                break
            boost = self.boosters.get(tokens[index])
            if not boost:
                continue
            scaled = boost * (1.0 - 0.05 * (distance - 1) * 3)
            delta += scaled if valence > 0 else -scaled
        return delta

    def _negation_factor(self, tokens: List[str], position: int) -> float:
        window = tokens[max(0, position - 3) : position]
        return NEGATION_SCALE if any(token in self.negations for token in window) else 1.0

    def _apply_contrast(self, tokens: List[str], valences: List[float]) -> List[float]:
        if "but" not in tokens:
            return valences
        pivot = tokens.index("but")
        scored_positions = [i for i, token in enumerate(tokens) if token in self.lexicon]
        adjusted = []
        for value, token_position in zip(valences, scored_positions):
            factor = CONTRAST_DAMPING if token_position < pivot else 1.0 + CONTRAST_DAMPING
            adjusted.append(value * factor)
        return adjusted

    def _punctuation_emphasis(self, text: str, polarity: float) -> float:
        exclamations = min(text.count("!"), 4) * EXCLAMATION_BOOST
        questions = text.count("?")
        question_emphasis = 0.0
        if questions > 1:
            question_emphasis = QUESTION_BOOST * min(questions, 3)
        emphasis = exclamations + question_emphasis
        return emphasis if polarity >= 0 else -emphasis


class SentimentEngine:
    """Public entry point; prefers vaderSentiment, falls back to the local one."""

    def __init__(
        self, prefer_vader: bool = True, window: int = 12, language: str = "en"
    ) -> None:
        self.language = language
        self._analyzers: Dict[str, LexiconSentimentAnalyzer] = {}
        self._fallback = self._analyzer_for(language)
        self._vader = None
        self.engine_name = self._fallback.engine
        if prefer_vader and language == "en":
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

                self._vader = SentimentIntensityAnalyzer()
                self.engine_name = "vader"
            except Exception:
                self._vader = None
        self._history: Deque[float] = deque(maxlen=max(2, window))

    def _analyzer_for(self, language: str) -> LexiconSentimentAnalyzer:
        if language not in self._analyzers:
            self._analyzers[language] = LexiconSentimentAnalyzer(language=language)
        return self._analyzers[language]

    def score(self, text: str, language: Optional[str] = None) -> SentimentScore:
        if language and language != self.language:
            self.language = language
            self._fallback = self._analyzer_for(language)
            if language != "en":
                self._vader = None
                self.engine_name = f"lexiflow-lexicon:{language}"

        if self._vader is not None:
            raw = self._vader.polarity_scores(text or "")
            result = SentimentScore(
                compound=round(float(raw["compound"]), 4),
                positive=round(float(raw["pos"]), 4),
                negative=round(float(raw["neg"]), 4),
                neutral=round(float(raw["neu"]), 4),
                label=label_for(float(raw["compound"])),
                engine="vader",
            )
        else:
            result = self._fallback.polarity_scores(text)
        self._history.append(result.compound)
        return result

    @property
    def rolling_average(self) -> float:
        if not self._history:
            return 0.0
        return round(sum(self._history) / len(self._history), 4)

    def momentum(self) -> float:
        """Difference between the newest half and the oldest half of the window."""
        if len(self._history) < 4:
            return 0.0
        values = list(self._history)
        midpoint = len(values) // 2
        older = sum(values[:midpoint]) / midpoint
        newer = sum(values[midpoint:]) / (len(values) - midpoint)
        return round(newer - older, 4)

    def history(self) -> List[float]:
        return list(self._history)

    def extend_lexicon(self, entries: Iterable[tuple[str, float]]) -> None:
        self._fallback.lexicon.update({word.lower(): float(value) for word, value in entries})
