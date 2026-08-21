from core.video_i18n import (
    normalize_language,
    resolve_video_language,
    tr,
)


def test_language_aliases():
    assert normalize_language("English") == "en"
    assert normalize_language("Türkçe") == "tr"
    assert normalize_language("Español") == "es"


def test_noninteractive_default():
    assert (
        resolve_video_language(
            None,
            interactive=False,
            default="de",
        )
        == "de"
    )


def test_translation_formatting():
    text = tr(
        "en",
        "actual_pass_to",
        receiver_id=7,
    )

    assert "ID 7" in text
    assert "pass" in text.lower()


def test_german_title():
    assert (
        tr(
            "de",
            "decision_moment",
        )
        == "ENTSCHEIDUNGSMOMENT"
    )
