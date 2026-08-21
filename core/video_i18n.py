from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_LANGUAGES = {
    "tr": "Türkçe",
    "en": "English",
    "es": "Español",
    "de": "Deutsch",
    "fr": "Français",
    "pt": "Português",
}


_TRANSLATIONS = {
    "tr": {
        "defense_view": "SAVUNMA BAKIŞI",
        "attack_view": "HÜCUM BAKIŞI",
        "decision_moment": "KARAR ANI",
        "alternative_defense": "ALTERNATİF SAVUNMA",
        "real_action": "GERÇEK AKSİYON",
        "attack_merit": "Hücum Kalitesi",
        "defense_vulnerability": "Savunma Zafiyeti",
        "model_top_option": "Sistemin en yüksek puanlı opsiyonu",
        "actual_decision": "Gerçek karar",
        "actual_pass_to": "Gerçek karar: ID {receiver_id}'ye pas",
        "actual_pass": "Gerçek karar: pas",
        "actual_shot": "Gerçek karar: ŞUT",
        "actual_carry": "Gerçek karar: topu taşıma / oyuna devam",
        "actual_unknown": "Gerçek karar: güvenilir biçimde çözülemedi",
        "no_ranked_option": "Sistem: güvenilir BEST/GOOD pas bulunamadı",
        "ranked_option": (
            "Sistemin en yüksek puanlı opsiyonu: "
            "ID {receiver_id} | {category} | skor={score}"
        ),
        "match_best": (
            "Gerçek karar, sistemin en yüksek puanlı seçeneğiyle eşleşti."
        ),
        "chose_alternative": (
            "Sistem ID {best_id} seçeneğini daha yüksek sıralarken "
            "gerçekte ID {actual_id} tercih edildi."
        ),
        "shot_over_pass": (
            "Sistem ID {best_id} yönünde güçlü bir pas alternatifi görürken "
            "top sahibi şutu tercih etti."
        ),
        "no_clear_comparison": (
            "Gerçek aksiyon güvenilir biçimde karşılaştırılamadı."
        ),
        "tactical_alternative_note": (
            "Bu bir taktik alternatifidir; kesin nedensel iddia değildir."
        ),
        "position_continues": "Pozisyonun gerçek devamı.",
        "outcome_shot": "Sekans şutla sonuçlanıyor.",
        "select_language": "Video dili seçin:",
        "invalid_language": "Geçersiz seçim. Tekrar deneyin.",
    },

    "en": {
        "defense_view": "DEFENSIVE VIEW",
        "attack_view": "ATTACKING VIEW",
        "decision_moment": "DECISION MOMENT",
        "alternative_defense": "DEFENSIVE ALTERNATIVE",
        "real_action": "ACTUAL PLAY",
        "attack_merit": "Attack Merit",
        "defense_vulnerability": "Defensive Vulnerability",
        "model_top_option": "Model's highest-ranked option",
        "actual_decision": "Actual decision",
        "actual_pass_to": "Actual decision: pass to ID {receiver_id}",
        "actual_pass": "Actual decision: pass",
        "actual_shot": "Actual decision: SHOT",
        "actual_carry": "Actual decision: carry / continue play",
        "actual_unknown": "Actual decision: could not be resolved reliably",
        "no_ranked_option": "Model: no reliable BEST/GOOD passing option",
        "ranked_option": (
            "Model's highest-ranked option: "
            "ID {receiver_id} | {category} | score={score}"
        ),
        "match_best": (
            "The actual decision matched the model's highest-ranked option."
        ),
        "chose_alternative": (
            "The model ranked ID {best_id} higher, while the actual choice "
            "was ID {actual_id}."
        ),
        "shot_over_pass": (
            "The model identified a strong passing alternative toward ID "
            "{best_id}, while the player chose to shoot."
        ),
        "no_clear_comparison": (
            "The actual action could not be compared reliably."
        ),
        "tactical_alternative_note": (
            "This is a tactical alternative, not a definitive causal claim."
        ),
        "position_continues": "Actual continuation of the play.",
        "outcome_shot": "The sequence ends with a shot.",
        "select_language": "Select video language:",
        "invalid_language": "Invalid selection. Please try again.",
    },

    "es": {
        "defense_view": "PERSPECTIVA DEFENSIVA",
        "attack_view": "PERSPECTIVA OFENSIVA",
        "decision_moment": "MOMENTO DE DECISIÓN",
        "alternative_defense": "ALTERNATIVA DEFENSIVA",
        "real_action": "JUGADA REAL",
        "attack_merit": "Mérito ofensivo",
        "defense_vulnerability": "Vulnerabilidad defensiva",
        "model_top_option": "Opción mejor valorada por el sistema",
        "actual_decision": "Decisión real",
        "actual_pass_to": "Decisión real: pase al ID {receiver_id}",
        "actual_pass": "Decisión real: pase",
        "actual_shot": "Decisión real: DISPARO",
        "actual_carry": "Decisión real: conducción / continuidad",
        "actual_unknown": "Decisión real: no pudo resolverse con fiabilidad",
        "no_ranked_option": "Sistema: no hay una opción BEST/GOOD fiable",
        "ranked_option": (
            "Opción mejor valorada por el sistema: "
            "ID {receiver_id} | {category} | puntuación={score}"
        ),
        "match_best": (
            "La decisión real coincide con la opción mejor valorada."
        ),
        "chose_alternative": (
            "El sistema valoró mejor al ID {best_id}, pero la elección real "
            "fue el ID {actual_id}."
        ),
        "shot_over_pass": (
            "El sistema detectó una alternativa de pase fuerte hacia el ID "
            "{best_id}, pero el jugador eligió disparar."
        ),
        "no_clear_comparison": (
            "La acción real no pudo compararse con fiabilidad."
        ),
        "tactical_alternative_note": (
            "Es una alternativa táctica, no una afirmación causal definitiva."
        ),
        "position_continues": "Continuación real de la jugada.",
        "outcome_shot": "La secuencia termina con un disparo.",
        "select_language": "Selecciona el idioma del video:",
        "invalid_language": "Selección no válida. Inténtalo de nuevo.",
    },

    "de": {
        "defense_view": "DEFENSIVE SICHT",
        "attack_view": "OFFENSIVE SICHT",
        "decision_moment": "ENTSCHEIDUNGSMOMENT",
        "alternative_defense": "DEFENSIVE ALTERNATIVE",
        "real_action": "TATSÄCHLICHE AKTION",
        "attack_merit": "Offensivqualität",
        "defense_vulnerability": "Defensive Anfälligkeit",
        "model_top_option": "Höchstbewertete Option des Systems",
        "actual_decision": "Tatsächliche Entscheidung",
        "actual_pass_to": "Tatsächliche Entscheidung: Pass zu ID {receiver_id}",
        "actual_pass": "Tatsächliche Entscheidung: Pass",
        "actual_shot": "Tatsächliche Entscheidung: SCHUSS",
        "actual_carry": "Tatsächliche Entscheidung: Dribbling / Fortsetzung",
        "actual_unknown": "Tatsächliche Entscheidung konnte nicht sicher ermittelt werden",
        "no_ranked_option": "System: keine verlässliche BEST/GOOD-Passoption",
        "ranked_option": (
            "Höchstbewertete Option des Systems: "
            "ID {receiver_id} | {category} | Score={score}"
        ),
        "match_best": (
            "Die tatsächliche Entscheidung entsprach der höchstbewerteten Option."
        ),
        "chose_alternative": (
            "Das System bewertete ID {best_id} höher, tatsächlich wurde "
            "ID {actual_id} gewählt."
        ),
        "shot_over_pass": (
            "Das System erkannte eine starke Passalternative zu ID {best_id}, "
            "der Spieler entschied sich jedoch für den Schuss."
        ),
        "no_clear_comparison": (
            "Die tatsächliche Aktion konnte nicht zuverlässig verglichen werden."
        ),
        "tactical_alternative_note": (
            "Dies ist eine taktische Alternative, keine definitive Kausalaussage."
        ),
        "position_continues": "Tatsächliche Fortsetzung der Aktion.",
        "outcome_shot": "Die Sequenz endet mit einem Schuss.",
        "select_language": "Videosprache auswählen:",
        "invalid_language": "Ungültige Auswahl. Bitte erneut versuchen.",
    },

    "fr": {
        "defense_view": "POINT DE VUE DÉFENSIF",
        "attack_view": "POINT DE VUE OFFENSIF",
        "decision_moment": "MOMENT DE DÉCISION",
        "alternative_defense": "ALTERNATIVE DÉFENSIVE",
        "real_action": "ACTION RÉELLE",
        "attack_merit": "Qualité offensive",
        "defense_vulnerability": "Vulnérabilité défensive",
        "model_top_option": "Option la mieux classée par le système",
        "actual_decision": "Décision réelle",
        "actual_pass_to": "Décision réelle : passe vers ID {receiver_id}",
        "actual_pass": "Décision réelle : passe",
        "actual_shot": "Décision réelle : TIR",
        "actual_carry": "Décision réelle : conduite / poursuite",
        "actual_unknown": "Décision réelle : impossible à déterminer avec fiabilité",
        "no_ranked_option": "Système : aucune option BEST/GOOD fiable",
        "ranked_option": (
            "Option la mieux classée par le système : "
            "ID {receiver_id} | {category} | score={score}"
        ),
        "match_best": (
            "La décision réelle correspond à l'option la mieux classée."
        ),
        "chose_alternative": (
            "Le système classait ID {best_id} plus haut, mais l'option réelle "
            "était ID {actual_id}."
        ),
        "shot_over_pass": (
            "Le système identifiait une forte option de passe vers ID {best_id}, "
            "mais le joueur a choisi de tirer."
        ),
        "no_clear_comparison": (
            "L'action réelle n'a pas pu être comparée de façon fiable."
        ),
        "tactical_alternative_note": (
            "Il s'agit d'une alternative tactique, pas d'une causalité certaine."
        ),
        "position_continues": "Suite réelle de l'action.",
        "outcome_shot": "La séquence se termine par un tir.",
        "select_language": "Choisissez la langue de la vidéo :",
        "invalid_language": "Sélection invalide. Réessayez.",
    },

    "pt": {
        "defense_view": "VISÃO DEFENSIVA",
        "attack_view": "VISÃO OFENSIVA",
        "decision_moment": "MOMENTO DA DECISÃO",
        "alternative_defense": "ALTERNATIVA DEFENSIVA",
        "real_action": "AÇÃO REAL",
        "attack_merit": "Mérito ofensivo",
        "defense_vulnerability": "Vulnerabilidade defensiva",
        "model_top_option": "Opção mais bem classificada pelo sistema",
        "actual_decision": "Decisão real",
        "actual_pass_to": "Decisão real: passe para ID {receiver_id}",
        "actual_pass": "Decisão real: passe",
        "actual_shot": "Decisão real: CHUTE",
        "actual_carry": "Decisão real: condução / continuidade",
        "actual_unknown": "Decisão real: não foi possível determinar com segurança",
        "no_ranked_option": "Sistema: nenhuma opção BEST/GOOD confiável",
        "ranked_option": (
            "Opção mais bem classificada pelo sistema: "
            "ID {receiver_id} | {category} | pontuação={score}"
        ),
        "match_best": (
            "A decisão real coincidiu com a opção mais bem classificada."
        ),
        "chose_alternative": (
            "O sistema classificou ID {best_id} acima, mas a escolha real "
            "foi ID {actual_id}."
        ),
        "shot_over_pass": (
            "O sistema identificou uma forte alternativa de passe para ID "
            "{best_id}, mas o jogador optou pelo chute."
        ),
        "no_clear_comparison": (
            "A ação real não pôde ser comparada com segurança."
        ),
        "tactical_alternative_note": (
            "Esta é uma alternativa tática, não uma afirmação causal definitiva."
        ),
        "position_continues": "Continuação real da jogada.",
        "outcome_shot": "A sequência termina com um chute.",
        "select_language": "Selecione o idioma do vídeo:",
        "invalid_language": "Seleção inválida. Tente novamente.",
    },
}


@dataclass(frozen=True)
class VideoLanguage:
    code: str
    name: str


def normalize_language(value: str | None) -> str | None:
    if value is None:
        return None

    value = str(value).strip().lower()

    aliases = {
        "turkish": "tr",
        "turkce": "tr",
        "türkçe": "tr",
        "english": "en",
        "spanish": "es",
        "espanol": "es",
        "español": "es",
        "german": "de",
        "deutsch": "de",
        "french": "fr",
        "français": "fr",
        "francais": "fr",
        "portuguese": "pt",
        "português": "pt",
        "portugues": "pt",
    }

    value = aliases.get(value, value)

    if value not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {value}. "
            f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
        )

    return value


def choose_language_interactively(
    *,
    default: str = "tr",
) -> str:
    default = normalize_language(default) or "tr"

    print()
    print(_TRANSLATIONS[default]["select_language"])

    items = list(SUPPORTED_LANGUAGES.items())

    for index, (code, name) in enumerate(items, start=1):
        marker = " *" if code == default else ""
        print(f"  {index}. {name} ({code}){marker}")

    print(
        f"Enter = {SUPPORTED_LANGUAGES[default]} ({default})"
    )

    while True:
        raw = input("> ").strip()

        if not raw:
            return default

        # Direct code/name.
        try:
            normalized = normalize_language(raw)
            if normalized is not None:
                return normalized
        except ValueError:
            pass

        # Menu number.
        if raw.isdigit():
            index = int(raw)

            if 1 <= index <= len(items):
                return items[index - 1][0]

        print(_TRANSLATIONS[default]["invalid_language"])


def resolve_video_language(
    value: str | None,
    *,
    interactive: bool = True,
    default: str = "tr",
) -> str:
    if value:
        return normalize_language(value) or default

    if interactive:
        return choose_language_interactively(
            default=default
        )

    return normalize_language(default) or "tr"


def tr(
    language: str,
    key: str,
    **kwargs,
) -> str:
    language = normalize_language(language) or "tr"

    table = _TRANSLATIONS.get(
        language,
        _TRANSLATIONS["tr"],
    )

    template = table.get(
        key,
        _TRANSLATIONS["tr"].get(
            key,
            key,
        ),
    )

    return str(template).format(**kwargs)
