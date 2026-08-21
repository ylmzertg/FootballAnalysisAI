# Video i18n ve Çoklu Dil Render Mimarisi

> Durum: Aktif ürün/mimari kararı  
> Tarih: 2026-08-21

## Amaç

FootballAnalysisAI analiz çekirdeği dile bağlı olmayacak. Aynı analiz sonucu yeniden hesaplanmadan farklı dillerde analyst/telestration veya anime video üretilebilecek.

```text
Gerçek maç videosu
      ↓
Football Analysis Engine
      ↓
Analyst Incident / Marking / Pass Options / Decision Comparison JSON
      ↓
Dil seçimi
      ↓
TR / EN / ES / DE / FR / PT render
```

## Desteklenen diller

İlk sürümde:

- `tr` — Türkçe
- `en` — English
- `es` — Español
- `de` — Deutsch
- `fr` — Français
- `pt` — Português

Yeni diller daha sonra aynı translation-table yapısına eklenebilir.

## Video oluşturulurken dil seçimi

Renderer iki çalışma biçimini desteklemeli.

### 1. İnteraktif kullanım

Kullanıcı `--language` parametresi vermezse renderer başlamadan önce dil sorulmalı:

```text
Video dili seçin:

1. Türkçe (tr)
2. English (en)
3. Español (es)
4. Deutsch (de)
5. Français (fr)
6. Português (pt)

Enter = Türkçe (tr)
>
```

### 2. Parametre ile otomatik kullanım

Örneğin:

```powershell
python -m scripts.render_analyst_v21 `
  --language en `
  ...
```

Bu durumda kullanıcıya soru sorulmadan İngilizce render alınır.

Pipeline/CI gibi interaktif olmayan kullanımlarda dil parametresi açıkça verilmelidir veya ileride config dosyasından okunmalıdır:

```yaml
render:
  language: en
```

## Dil, analiz verisinden ayrıdır

Aşağıdaki veriler dil-bağımsız kalmalıdır:

- `Analyst Incident JSON`
- `Decision Comparison JSON`
- `Marking JSONL`
- `Pass Options JSONL`
- `Shot/Event JSONL`
- `Storyboard JSON`

Örneğin analiz yalnız bir kez üretilebilir:

```text
INC-0002
Attack Merit = HIGH
Defense Vulnerability = HIGH
Best option = ID18
Actual action = PASS -> ID7
Shot = True
```

Sonrasında aynı veriden:

```text
analyst_tr.mp4
analyst_en.mp4
analyst_es.mp4
analyst_de.mp4
analyst_fr.mp4
analyst_pt.mp4
```

üretilebilir.

## Renderer metinleri

Hard-coded metin kullanılmamalıdır.

Örneğin:

```python
"SAVUNMA BAKIŞI"
```

yerine:

```python
tr(language, "defense_view")
```

kullanılmalıdır.

İlk translation anahtarları arasında:

- `defense_view`
- `attack_view`
- `decision_moment`
- `alternative_defense`
- `real_action`
- `attack_merit`
- `defense_vulnerability`
- `ranked_option`
- `actual_pass_to`
- `actual_shot`
- `actual_unknown`
- `match_best`
- `chose_alternative`
- `shot_over_pass`
- `tactical_alternative_note`

bulunmalıdır.

## Karar karşılaştırması terminolojisi

Mevcut Pass Options Ranking v1 açıklanabilir, heuristik/weighted bir modeldir. Bu nedenle:

```text
ÖNERİLEN PAS
```

yerine daha doğru olarak:

```text
Sistemin en yüksek puanlı opsiyonu
```

kullanılmalıdır.

İngilizce:

```text
Model's highest-ranked option
```

Bu, sistemin henüz learned xT/EPV-optimal action modeli olmadığını doğru biçimde yansıtır.

## Analyst Renderer için hedef sıra

```text
SAVUNMA BAKIŞI
      ↓
HÜCUM BAKIŞI
      ↓
KARAR ANI
      ↓
Sistemin en yüksek puanlı opsiyonu
VS
Gerçek karar
      ↓
ALTERNATİF SAVUNMA
      ↓
GERÇEK AKSİYON
```

Bu bölüm başlıkları ve açıklamaları seçilen dile göre render edilmelidir.

## Anime Renderer ile ortak kullanım

Anime/Tsubasa-esintili yeniden canlandırma ayrı bir analiz sistemi olmayacaktır. Aynı semantic incident/storyboard verisini kullanacaktır.

```text
Analysis Engine
   ├── Broadcast / Telestration Renderer
   └── Anime Reconstruction Renderer
```

Anime renderer da video üretmeden önce aynı dil seçimini kullanabilmelidir.

Dil seçimi ileride:

```text
language = en
  ↓
English overlay
English narration script
English TTS voice
English subtitle
English YouTube metadata template
```

akışını kontrol edebilir.

## Kod yerleşimi

Önerilen ortak modül:

```text
core/video_i18n.py
```

Temel API:

```python
resolve_video_language(...)
tr(language, key, **kwargs)
```

Renderer CLI:

```python
p.add_argument(
    "--language",
    default="",
    help="tr,en,es,de,fr,pt. Empty asks interactively.",
)
```

Başlangıç:

```python
language = resolve_video_language(
    args.language,
    interactive=True,
    default="tr",
)
```

## Taşınabilirlik

Dil tabloları repository içinde tutulmalıdır. Makineye özgü path veya kullanıcıya özgü ayar hard-code edilmemelidir.

Başka Windows bilgisayarda clone sonrası aynı renderer aynı dil seçenekleriyle çalışmalıdır.

## Sonraki geliştirme

1. `core/video_i18n.py` ortak i18n katmanını repo içinde standartlaştır.
2. `render_analyst_v21.py` içine `--language` + interaktif seçim ekle.
3. Overlay metinlerini translation key'lerine taşı.
4. Storyboard ve analiz JSON'larını dil-bağımsız tut.
5. Narration/TTS katmanına aynı language code'u geçir.
6. Anime renderer oluşturulduğunda aynı i18n API'yi kullan.
