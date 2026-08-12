"""Rule packs and valence lexicons for the non-English languages we support."""

from __future__ import annotations

import re
from typing import Dict, List

from .rules import RuleSpec

SPANISH_LEXICON: Dict[str, float] = {
    "excelente": 2.9, "genial": 2.7, "bueno": 1.9, "buena": 1.9, "mejor": 1.9, "perfecto": 3.0,
    "gracias": 2.2, "feliz": 2.6, "contento": 2.3, "encanta": 2.7, "increíble": 2.8,
    "fantástico": 3.0, "éxito": 2.7, "listo": 1.4, "resuelto": 2.0, "acuerdo": 1.5,
    "tranquilo": 1.5, "seguro": 1.7, "rápido": 1.4, "claro": 1.4, "útil": 2.0, "fuerte": 1.6,
    "malo": -2.5, "mala": -2.5, "peor": -2.4, "terrible": -2.8, "horrible": -2.9,
    "problema": -1.9, "error": -2.0, "fallo": -2.4, "falla": -2.4, "retraso": -1.8,
    "preocupado": -2.1, "preocupa": -2.0, "triste": -2.4, "enojado": -2.5, "molesto": -2.0,
    "difícil": -1.6, "imposible": -2.2, "riesgo": -1.6, "urgente": -1.4, "bloqueado": -2.0,
    "roto": -2.3, "lento": -1.6, "caro": -1.3, "perdido": -1.8, "confuso": -1.8, "miedo": -2.3,
}

FRENCH_LEXICON: Dict[str, float] = {
    "excellent": 2.9, "génial": 2.7, "bon": 1.9, "bonne": 1.9, "mieux": 1.9, "parfait": 3.0,
    "merci": 2.2, "heureux": 2.6, "content": 2.3, "adore": 2.7, "incroyable": 2.8,
    "fantastique": 3.0, "succès": 2.7, "prêt": 1.4, "résolu": 2.0, "accord": 1.5,
    "tranquille": 1.5, "sûr": 1.5, "rapide": 1.4, "clair": 1.4, "utile": 2.0, "solide": 1.8,
    "mauvais": -2.5, "pire": -2.4, "terrible": -2.8, "horrible": -2.9, "problème": -1.9,
    "erreur": -2.0, "panne": -2.4, "échec": -2.6, "retard": -1.8, "inquiet": -2.1,
    "triste": -2.4, "fâché": -2.5, "difficile": -1.6, "impossible": -2.2, "risque": -1.6,
    "urgent": -1.4, "bloqué": -2.0, "cassé": -2.3, "lent": -1.6, "cher": -1.3, "perdu": -1.8,
    "confus": -1.8, "peur": -2.3, "désolé": -1.3,
}

GERMAN_LEXICON: Dict[str, float] = {
    "ausgezeichnet": 2.9, "toll": 2.6, "gut": 1.9, "besser": 1.9, "perfekt": 3.0, "danke": 2.2,
    "glücklich": 2.6, "zufrieden": 2.3, "liebe": 2.7, "unglaublich": 2.5, "fantastisch": 3.0,
    "erfolg": 2.7, "fertig": 1.4, "gelöst": 2.0, "einig": 1.5, "ruhig": 1.4, "sicher": 1.7,
    "schnell": 1.4, "klar": 1.4, "nützlich": 2.0, "stark": 1.6, "super": 2.6,
    "schlecht": -2.5, "schlimmer": -2.4, "schrecklich": -2.8, "furchtbar": -2.9,
    "problem": -1.9, "fehler": -2.0, "ausfall": -2.4, "verzögerung": -1.8, "besorgt": -2.1,
    "traurig": -2.4, "wütend": -2.5, "schwierig": -1.6, "unmöglich": -2.2, "risiko": -1.6,
    "dringend": -1.4, "blockiert": -2.0, "kaputt": -2.3, "langsam": -1.6, "teuer": -1.3,
    "verloren": -1.8, "verwirrt": -1.8, "angst": -2.3, "leider": -1.4,
}

ITALIAN_LEXICON: Dict[str, float] = {
    "eccellente": 2.9, "ottimo": 2.7, "buono": 1.9, "buona": 1.9, "meglio": 1.9, "perfetto": 3.0,
    "grazie": 2.2, "felice": 2.6, "contento": 2.3, "adoro": 2.7, "incredibile": 2.6,
    "fantastico": 3.0, "successo": 2.7, "pronto": 1.4, "risolto": 2.0, "accordo": 1.5,
    "tranquillo": 1.5, "sicuro": 1.7, "veloce": 1.4, "chiaro": 1.4, "utile": 2.0, "forte": 1.6,
    "cattivo": -2.5, "peggio": -2.4, "terribile": -2.8, "orribile": -2.9, "problema": -1.9,
    "errore": -2.0, "guasto": -2.4, "ritardo": -1.8, "preoccupato": -2.1, "triste": -2.4,
    "arrabbiato": -2.5, "difficile": -1.6, "impossibile": -2.2, "rischio": -1.6,
    "urgente": -1.4, "bloccato": -2.0, "rotto": -2.3, "lento": -1.6, "caro": -1.3,
    "perso": -1.8, "confuso": -1.8, "paura": -2.3, "scusa": -1.2,
}

PORTUGUESE_LEXICON: Dict[str, float] = {
    "excelente": 2.9, "ótimo": 2.7, "bom": 1.9, "boa": 1.9, "melhor": 1.9, "perfeito": 3.0,
    "obrigado": 2.2, "feliz": 2.6, "contente": 2.3, "adoro": 2.7, "incrível": 2.8,
    "fantástico": 3.0, "sucesso": 2.7, "pronto": 1.4, "resolvido": 2.0, "acordo": 1.5,
    "tranquilo": 1.5, "seguro": 1.7, "rápido": 1.4, "claro": 1.4, "útil": 2.0, "forte": 1.6,
    "ruim": -2.5, "pior": -2.4, "terrível": -2.8, "horrível": -2.9, "problema": -1.9,
    "erro": -2.0, "falha": -2.4, "atraso": -1.8, "preocupado": -2.1, "triste": -2.4,
    "irritado": -2.5, "difícil": -1.6, "impossível": -2.2, "risco": -1.6, "urgente": -1.4,
    "bloqueado": -2.0, "quebrado": -2.3, "lento": -1.6, "caro": -1.3, "perdido": -1.8,
    "confuso": -1.8, "medo": -2.3, "desculpa": -1.2,
}

LEXICONS: Dict[str, Dict[str, float]] = {
    "es": SPANISH_LEXICON,
    "fr": FRENCH_LEXICON,
    "de": GERMAN_LEXICON,
    "it": ITALIAN_LEXICON,
    "pt": PORTUGUESE_LEXICON,
}

NEGATIONS: Dict[str, frozenset] = {
    "es": frozenset({"no", "nunca", "nada", "ni", "jamás", "tampoco", "sin"}),
    "fr": frozenset({"ne", "pas", "jamais", "rien", "aucun", "sans", "ni"}),
    "de": frozenset({"nicht", "kein", "keine", "keinen", "nie", "niemals", "ohne", "nichts"}),
    "it": frozenset({"non", "mai", "niente", "nessuno", "senza", "né"}),
    "pt": frozenset({"não", "nunca", "nada", "nenhum", "sem", "nem", "jamais"}),
}

BOOSTERS: Dict[str, Dict[str, float]] = {
    "es": {"muy": 0.293, "súper": 0.293, "bastante": 0.193, "poco": -0.293, "algo": -0.193},
    "fr": {"très": 0.293, "vraiment": 0.293, "assez": 0.193, "peu": -0.293, "un_peu": -0.293},
    "de": {"sehr": 0.293, "wirklich": 0.293, "ziemlich": 0.193, "kaum": -0.293, "etwas": -0.193},
    "it": {"molto": 0.293, "davvero": 0.293, "abbastanza": 0.193, "poco": -0.293},
    "pt": {"muito": 0.293, "realmente": 0.293, "bastante": 0.193, "pouco": -0.293},
}

STOPWORDS: Dict[str, frozenset] = {
    "es": frozenset(
        """de la que el en y a los se del las un por con no una su para es al lo como más
        pero sus le ya o este sí porque esta entre cuando muy sin sobre también me hasta hay
        donde quien desde todo nos durante todos uno les ni contra otros ese eso ante ellos
        e esto mí antes algunos qué unos yo otro otras otra él tanto esa estos mucho""".split()
    ),
    "fr": frozenset(
        """de le la et les des en un une du dans est que pour qui sur pas plus par au ce il
        ne se ces cette nous vous ils elle son sa ses leur avec mais ou donc or ni car y a
        été être avoir fait tout tous comme si bien sans sous entre après avant encore""".split()
    ),
    "de": frozenset(
        """der die das und in den von zu mit sich des auf für ist im dem nicht ein eine als
        auch es an werden aus er hat dass sie nach wird bei einer um am sind noch wie einem
        über einen so zum haben nur oder aber vor bis mehr durch man sein wurde sei""".split()
    ),
    "it": frozenset(
        """di che il la e in un per una non con sono mi ma come questo più anche molto quando
        perché cosa tutto solo adesso noi loro essere fare avere dove chi tempo anno giorno
        sempre mai ancora dopo prima nel del alla dei delle degli""".split()
    ),
    "pt": frozenset(
        """de que não para com uma você por mais isso está muito como mas quando porque então
        aqui agora sim nós eles fazer ter ser tempo ano dia sempre nunca ainda depois antes
        tudo nada algo outro esse essa dos das nas nos pelo pela""".split()
    ),
}


def _compile(expression: str) -> re.Pattern[str]:
    return re.compile(expression, re.IGNORECASE)


SPANISH_RULES: List[RuleSpec] = [
    RuleSpec(
        "recordatorio",
        "action_item",
        _compile(r"\brecu[ée]rdame\s+(?:que\s+)?(.+?)(?=[.?!;]|$)"),
        0.95,
    ),
    RuleSpec(
        "compromiso",
        "action_item",
        _compile(r"\b(?:voy a|tengo que|hay que|necesitamos)\s+(.+?)(?=[.?!;]|$)"),
        0.85,
    ),
    RuleSpec(
        "peticion",
        "action_item",
        _compile(r"\b(?:puedes|podr[íi]as)\s+(?:por favor\s+)?(.+?)(?=[.?!;]|$)"),
        0.8,
    ),
    RuleSpec(
        "fecha_limite",
        "deadline",
        _compile(r"\b(?:fecha l[íi]mite|plazo)\s+(?:es|era)?\s*(.+?)(?=[.?!;,]|\s+y\b|$)"),
        0.95,
    ),
    RuleSpec(
        "entrega",
        "deadline",
        _compile(r"\b(?:para|antes del|antes de)\s+(lunes|martes|mi[ée]rcoles|jueves|viernes|"
                 r"s[áa]bado|domingo|ma[ñn]ana|hoy|fin de semana)\b"),
        0.85,
    ),
    RuleSpec(
        "bloqueo",
        "blocker",
        _compile(r"\b(?:estoy bloqueado|bloqueado (?:en|por)|atascado en)\s+(.+?)(?=[.?!;]|$)"),
        0.9,
    ),
    RuleSpec(
        "decision",
        "decision",
        _compile(r"\b(?:hemos decidido|decidimos)\s+(?:que\s+)?(.+?)(?=[.?!;]|$)"),
        0.9,
    ),
]

FRENCH_RULES: List[RuleSpec] = [
    RuleSpec(
        "rappel",
        "action_item",
        _compile(r"\brappelle[- ]moi\s+(?:de\s+)?(.+?)(?=[.?!;]|$)"),
        0.95,
    ),
    RuleSpec(
        "engagement",
        "action_item",
        _compile(r"\b(?:je vais|il faut|nous devons|on doit)\s+(.+?)(?=[.?!;]|$)"),
        0.85,
    ),
    RuleSpec(
        "demande",
        "action_item",
        _compile(
            r"\b(?:peux-tu|pouvez-vous|pourrais-tu)\s+(?:s'il te pla[îi]t\s+)?(.+?)(?=[.?!;]|$)"
        ),
        0.8,
    ),
    RuleSpec(
        "echeance",
        "deadline",
        _compile(r"\b(?:date limite|[ée]ch[ée]ance|d[ée]lai)\s+(?:est|était)?\s*(.+?)"
                 r"(?=[.?!;,]|\s+et\b|$)"),
        0.95,
    ),
    RuleSpec(
        "livraison",
        "deadline",
        _compile(r"\b(?:avant|pour)\s+(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|"
                 r"demain|aujourd'hui|la fin de la semaine)\b"),
        0.85,
    ),
    RuleSpec(
        "blocage",
        "blocker",
        _compile(
            r"\b(?:je suis bloqu[ée]|bloqu[ée] (?:sur|par)|coinc[ée] sur)\s+(.+?)(?=[.?!;]|$)"
        ),
        0.9,
    ),
    RuleSpec(
        "decision",
        "decision",
        _compile(r"\b(?:nous avons d[ée]cid[ée]|on a d[ée]cid[ée])\s+(?:de\s+|que\s+)?(.+?)"
                 r"(?=[.?!;]|$)"),
        0.9,
    ),
]

GERMAN_RULES: List[RuleSpec] = [
    RuleSpec(
        "erinnerung",
        "action_item",
        _compile(r"\berinnere mich\s+(?:daran,?\s+)?(.+?)(?=[.?!;]|$)"),
        0.95,
    ),
    RuleSpec(
        "zusage",
        "action_item",
        _compile(r"\b(?:ich werde|wir m[üu]ssen|ich muss|wir sollten)\s+(.+?)(?=[.?!;]|$)"),
        0.85,
    ),
    RuleSpec(
        "bitte",
        "action_item",
        _compile(r"\b(?:kannst du|k[öo]nnen sie|k[öo]nntest du)\s+(?:bitte\s+)?(.+?)(?=[.?!;]|$)"),
        0.8,
    ),
    RuleSpec(
        "frist",
        "deadline",
        _compile(r"\b(?:frist|deadline|abgabetermin)\s+(?:ist|war)?\s*(.+?)"
                 r"(?=[.?!;,]|\s+und\b|$)"),
        0.95,
    ),
    RuleSpec(
        "termin",
        "deadline",
        _compile(r"\b(?:bis)\s+(montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag|"
                 r"morgen|heute|ende der woche)\b"),
        0.85,
    ),
    RuleSpec(
        "blocker",
        "blocker",
        _compile(
            r"\b(?:ich bin blockiert|blockiert (?:durch|von)|h[äa]nge an)\s+(.+?)(?=[.?!;]|$)"
        ),
        0.9,
    ),
    RuleSpec(
        "entscheidung",
        "decision",
        _compile(r"\bwir haben entschieden,?\s+(?:dass\s+)?(.+?)(?=[.?!;]|$)"),
        0.9,
    ),
]

ITALIAN_RULES: List[RuleSpec] = [
    RuleSpec(
        "promemoria",
        "action_item",
        _compile(r"\bricordami\s+(?:di\s+)?(.+?)(?=[.?!;]|$)"),
        0.95,
    ),
    RuleSpec(
        "impegno",
        "action_item",
        _compile(r"\b(?:devo|dobbiamo|bisogna)\s+(.+?)(?=[.?!;]|$)"),
        0.85,
    ),
    RuleSpec(
        "richiesta",
        "action_item",
        _compile(r"\b(?:puoi|potresti|potete)\s+(?:per favore\s+)?(.+?)(?=[.?!;]|$)"),
        0.8,
    ),
    RuleSpec(
        "scadenza",
        "deadline",
        _compile(r"\b(?:scadenza|termine)\s+(?:è|era)?\s*(.+?)(?=[.?!;,]|\s+e\b|$)"),
        0.95,
    ),
    RuleSpec(
        "consegna",
        "deadline",
        _compile(r"\b(?:entro|prima di)\s+(luned[ìi]|marted[ìi]|mercoled[ìi]|gioved[ìi]|"
                 r"venerd[ìi]|sabato|domenica|domani|oggi|fine settimana)\b"),
        0.85,
    ),
    RuleSpec(
        "blocco",
        "blocker",
        _compile(r"\b(?:sono bloccato|bloccato (?:su|da))\s+(.+?)(?=[.?!;]|$)"),
        0.9,
    ),
    RuleSpec(
        "decisione",
        "decision",
        _compile(r"\b(?:abbiamo deciso|si è deciso)\s+(?:di\s+|che\s+)?(.+?)(?=[.?!;]|$)"),
        0.9,
    ),
]

PORTUGUESE_RULES: List[RuleSpec] = [
    RuleSpec(
        "lembrete",
        "action_item",
        _compile(r"\blembre?-?me\s+(?:de\s+)?(.+?)(?=[.?!;]|$)"),
        0.95,
    ),
    RuleSpec(
        "compromisso",
        "action_item",
        _compile(r"\b(?:vou|precisamos|preciso|temos que)\s+(.+?)(?=[.?!;]|$)"),
        0.85,
    ),
    RuleSpec(
        "pedido",
        "action_item",
        _compile(r"\b(?:voc[êe] pode|poderia)\s+(?:por favor\s+)?(.+?)(?=[.?!;]|$)"),
        0.8,
    ),
    RuleSpec(
        "prazo",
        "deadline",
        _compile(r"\b(?:prazo|data limite)\s+(?:é|era)?\s*(.+?)(?=[.?!;,]|\s+e\b|$)"),
        0.95,
    ),
    RuleSpec(
        "entrega",
        "deadline",
        _compile(r"\b(?:at[ée]|antes de)\s+(segunda|ter[çc]a|quarta|quinta|sexta|s[áa]bado|"
                 r"domingo|amanh[ãa]|hoje|fim de semana)\b"),
        0.85,
    ),
    RuleSpec(
        "bloqueio",
        "blocker",
        _compile(r"\b(?:estou bloqueado|bloqueado (?:em|por)|travado em)\s+(.+?)(?=[.?!;]|$)"),
        0.9,
    ),
    RuleSpec(
        "decisao",
        "decision",
        _compile(r"\b(?:decidimos|n[óo]s decidimos)\s+(?:que\s+)?(.+?)(?=[.?!;]|$)"),
        0.9,
    ),
]

RULE_PACKS: Dict[str, List[RuleSpec]] = {
    "es": SPANISH_RULES,
    "fr": FRENCH_RULES,
    "de": GERMAN_RULES,
    "it": ITALIAN_RULES,
    "pt": PORTUGUESE_RULES,
}


def rules_for(code: str) -> List[RuleSpec]:
    return RULE_PACKS.get(code, [])


def lexicon_for(code: str) -> Dict[str, float]:
    return LEXICONS.get(code, {})
