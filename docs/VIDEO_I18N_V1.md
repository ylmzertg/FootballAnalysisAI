# Video i18n v1 — Çoklu Dil Katmanı

## Desteklenen diller

```text
tr  Türkçe
en  English
es  Español
de  Deutsch
fr  Français
pt  Português
```

## Kullanım modeli

### 1. Komutta dil ver

```powershell
python -m scripts.render_analyst_v21 `
  --language en `
  ...
```

Bu durumda soru sormadan İngilizce üretir.

### 2. Dil parametresi verme

```powershell
python -m scripts.render_analyst_v21 `
  ...
```

Renderer başlamadan önce:

```text
Video dili seçin:
  1. Türkçe (tr) *
  2. English (en)
  3. Español (es)
  4. Deutsch (de)
  5. Français (fr)
  6. Português (pt)

Enter = Türkçe (tr)
>
```

gösterilir.

## Otomasyon / Pipeline

CI veya tam otomatik pipeline'da interaktif soru istenmez.

Bu durumda:

```text
--language en
```

zorunlu verilebilir veya ileride pipeline config içindeki:

```yaml
render:
  language: en
```

kullanılabilir.

## Mimari prensip

Analiz verisinin kendisi dile bağlı olmamalıdır.

```text
Analyst Incident JSON
Decision Comparison JSON
Marking JSON
        ↓
language-neutral semantic data
        ↓
i18n templates
        ↓
TR / EN / ES / DE / FR / PT video
```

Aynı incident bir kez analiz edilip altı dilde yeniden render edilebilir.

## Seslendirme

İleride aynı language code:

```text
language=en
```

şunları birlikte kontrol edebilir:

- overlay metinleri;
- narration script;
- ElevenLabs / TTS voice mapping;
- altyazılar;
- YouTube title/description templates.

## Renderer entegrasyonu

Renderer `parse_args()` içine:

```python
p.add_argument(
    "--language",
    default="",
    help="tr,en,es,de,fr,pt. Empty asks interactively.",
)
```

eklenir.

`main()` başında:

```python
from core.video_i18n import resolve_video_language, tr

language = resolve_video_language(
    args.language,
    interactive=True,
    default="tr",
)
```

Hard-coded:

```python
"SAVUNMA BAKIŞI"
```

yerine:

```python
tr(language, "defense_view")
```

kullanılır.

Aynı yöntem Anime Renderer için de kullanılacaktır.
