"""Extractive summarisation, keyphrase ranking and topic drift, all pure math."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'\-]+")

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be because been before
    being below between both but by can cannot could couldn't did didn't do does doesn't doing
    don't down during each few for from further had hadn't has hasn't have haven't having he he'd
    he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in
    into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once
    only or other ought our ours ourselves out over own same shan't she she'd she'll she's should
    shouldn't so some such than that that's the their theirs them themselves then there there's
    these they they'd they'll they're they've this those through to too under until up very was
    wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while
    who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours
    yourself yourselves just really actually basically kind sort like okay yeah yes um uh gonna
    wanna going get got know think mean say said thing things stuff lot bit right well now also
    """.split()
)

FILLER_TOKENS = frozenset({"um", "uh", "erm", "hmm", "mm", "ah", "eh", "like", "yeah", "okay"})


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    tokens = [match.group(0).lower() for match in WORD_PATTERN.finditer(text or "")]
    if not drop_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def split_sentences(text: str) -> List[str]:
    parts = [part.strip() for part in SENTENCE_SPLIT.split(text or "") if part and part.strip()]
    return parts


@dataclass
class Keyphrase:
    text: str
    score: float
    count: int = 1

    def as_dict(self) -> Dict[str, object]:
        return {"text": self.text, "score": round(self.score, 4), "count": self.count}


@dataclass
class SummarySentence:
    text: str
    score: float
    position: int

    def as_dict(self) -> Dict[str, object]:
        return {"text": self.text, "score": round(self.score, 4), "position": self.position}


@dataclass
class TopicShift:
    at_index: int
    similarity: float
    previous_keywords: List[str] = field(default_factory=list)
    current_keywords: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "at_index": self.at_index,
            "similarity": round(self.similarity, 4),
            "previous_keywords": list(self.previous_keywords),
            "current_keywords": list(self.current_keywords),
        }


class KeyphraseRanker:
    """RAKE: score candidate phrases by word degree over word frequency."""

    def __init__(self, max_phrase_words: int = 4, min_word_length: int = 3) -> None:
        self.max_phrase_words = max_phrase_words
        self.min_word_length = min_word_length

    def candidates(self, text: str) -> List[List[str]]:
        phrases: List[List[str]] = []
        for sentence in split_sentences(text):
            current: List[str] = []
            for token in re.split(r"[^A-Za-z'\-]+", sentence):
                lowered = token.lower()
                is_stop = (
                    not lowered
                    or lowered in STOPWORDS
                    or lowered in FILLER_TOKENS
                    or len(lowered) < self.min_word_length
                )
                if is_stop:
                    if current:
                        phrases.append(current)
                        current = []
                    continue
                current.append(lowered)
                if len(current) >= self.max_phrase_words:
                    phrases.append(current)
                    current = []
            if current:
                phrases.append(current)
        return phrases

    def rank(self, text: str, limit: int = 10) -> List[Keyphrase]:
        phrases = self.candidates(text)
        if not phrases:
            return []

        frequency: Counter[str] = Counter()
        degree: Counter[str] = Counter()
        for phrase in phrases:
            span = len(phrase) - 1
            for word in phrase:
                frequency[word] += 1
                degree[word] += span

        word_score = {
            word: (degree[word] + frequency[word]) / frequency[word] for word in frequency
        }

        totals: Dict[str, float] = defaultdict(float)
        counts: Counter[str] = Counter()
        for phrase in phrases:
            joined = " ".join(phrase)
            totals[joined] += sum(word_score[word] for word in phrase)
            counts[joined] += 1

        ranked = [
            Keyphrase(text=phrase, score=total / counts[phrase], count=counts[phrase])
            for phrase, total in totals.items()
        ]
        ranked.sort(key=lambda item: (-item.score * math.log1p(item.count), item.text))
        return ranked[:limit]


class TextRankSummarizer:
    """Sentence graph ranked by power iteration over normalised word overlap."""

    def __init__(self, damping: float = 0.85, iterations: int = 40, tolerance: float = 1e-5):
        self.damping = damping
        self.iterations = iterations
        self.tolerance = tolerance

    @staticmethod
    def _similarity(left: Sequence[str], right: Sequence[str]) -> float:
        if not left or not right:
            return 0.0
        overlap = len(set(left) & set(right))
        if overlap == 0:
            return 0.0
        denominator = math.log(len(left) + 1) + math.log(len(right) + 1)
        return overlap / denominator if denominator else 0.0

    def rank(self, sentences: Sequence[str]) -> List[SummarySentence]:
        if not sentences:
            return []
        if len(sentences) == 1:
            return [SummarySentence(sentences[0], 1.0, 0)]

        tokenized = [tokenize(sentence) for sentence in sentences]
        size = len(sentences)
        weights = [[0.0] * size for _ in range(size)]
        for row in range(size):
            for column in range(row + 1, size):
                score = self._similarity(tokenized[row], tokenized[column])
                weights[row][column] = score
                weights[column][row] = score

        row_sums = [sum(row) or 1.0 for row in weights]
        scores = [1.0 / size] * size
        for _ in range(self.iterations):
            updated = []
            for index in range(size):
                inbound = sum(
                    weights[other][index] / row_sums[other]
                    for other in range(size)
                    if other != index
                )
                updated.append((1.0 - self.damping) / size + self.damping * inbound * scores[index])
            total = sum(updated) or 1.0
            updated = [value / total for value in updated]
            if max(abs(a - b) for a, b in zip(updated, scores)) < self.tolerance:
                scores = updated
                break
            scores = updated

        ranked = [
            SummarySentence(text=sentences[index], score=scores[index], position=index)
            for index in range(size)
        ]
        ranked.sort(key=lambda item: -item.score)
        return ranked

    def summarize(self, sentences: Sequence[str], limit: int = 5) -> List[SummarySentence]:
        chosen = self.rank(sentences)[:limit]
        chosen.sort(key=lambda item: item.position)
        return chosen


class TopicTracker:
    """Detects conversational subject changes by cosine distance between windows."""

    def __init__(self, window: int = 6, threshold: float = 0.12, min_tokens: int = 8) -> None:
        self.window = window
        self.threshold = threshold
        self.min_tokens = min_tokens
        self._lines: List[List[str]] = []
        self._shifts: List[TopicShift] = []

    @staticmethod
    def _cosine(left: Counter, right: Counter) -> float:
        if not left or not right:
            return 0.0
        shared = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in shared)
        if numerator == 0:
            return 0.0
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return numerator / (left_norm * right_norm)

    def push(self, text: str) -> Optional[TopicShift]:
        tokens = tokenize(text)
        self._lines.append(tokens)
        if len(self._lines) < self.window * 2:
            return None

        window = self.window
        previous = Counter(
            token for line in self._lines[-window * 2 : -window] for token in line
        )
        current = Counter(token for line in self._lines[-self.window :] for token in line)
        if sum(previous.values()) < self.min_tokens or sum(current.values()) < self.min_tokens:
            return None

        similarity = self._cosine(previous, current)
        if similarity >= self.threshold:
            return None
        if self._shifts and len(self._lines) - self._shifts[-1].at_index < self.window:
            return None

        shift = TopicShift(
            at_index=len(self._lines),
            similarity=similarity,
            previous_keywords=[token for token, _ in previous.most_common(5)],
            current_keywords=[token for token, _ in current.most_common(5)],
        )
        self._shifts.append(shift)
        return shift

    @property
    def shifts(self) -> List[TopicShift]:
        return list(self._shifts)


@dataclass
class ConversationDigest:
    summary: List[SummarySentence] = field(default_factory=list)
    keyphrases: List[Keyphrase] = field(default_factory=list)
    topics: List[TopicShift] = field(default_factory=list)
    word_count: int = 0
    unique_words: int = 0
    speaking_rate: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "summary": [item.as_dict() for item in self.summary],
            "keyphrases": [item.as_dict() for item in self.keyphrases],
            "topics": [item.as_dict() for item in self.topics],
            "word_count": self.word_count,
            "unique_words": self.unique_words,
            "speaking_rate": round(self.speaking_rate, 2),
        }

    def as_markdown(self) -> str:
        lines = ["## Summary", ""]
        if self.summary:
            lines.extend(f"- {item.text}" for item in self.summary)
        else:
            lines.append("- nothing captured yet")
        lines.extend(["", "## Key topics", ""])
        if self.keyphrases:
            lines.extend(f"- {item.text} ({item.score:.1f})" for item in self.keyphrases)
        else:
            lines.append("- none")
        return "\n".join(lines)


class DigestBuilder:
    """One call turns a whole session's transcript into a shareable digest."""

    def __init__(self, summary_limit: int = 5, keyphrase_limit: int = 12) -> None:
        self.summary_limit = summary_limit
        self.keyphrase_limit = keyphrase_limit
        self.summarizer = TextRankSummarizer()
        self.ranker = KeyphraseRanker()
        self.topics = TopicTracker()

    def build(
        self,
        lines: Iterable[str],
        audio_seconds: float = 0.0,
        topics: Optional[List[TopicShift]] = None,
    ) -> ConversationDigest:
        collected = [line.strip() for line in lines if line and line.strip()]
        if not collected:
            return ConversationDigest()

        joined = " ".join(collected)
        sentences: List[str] = []
        for line in collected:
            sentences.extend(split_sentences(line) or [line])

        words = tokenize(joined, drop_stopwords=False)
        return ConversationDigest(
            summary=self.summarizer.summarize(sentences, self.summary_limit),
            keyphrases=self.ranker.rank(joined, self.keyphrase_limit),
            topics=list(topics or []),
            word_count=len(words),
            unique_words=len(set(words)),
            speaking_rate=(len(words) / (audio_seconds / 60.0)) if audio_seconds > 0 else 0.0,
        )
