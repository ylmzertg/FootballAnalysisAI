# Pass Options Ranking v1

Concept 1 için ilk gerçek pas-karar katmanı.

Eski Tactical Engine:

```text
OPEN / BLOCKED
```

Yeni ranking:

```text
BEST
GOOD
RISKY
BLOCKED
```

## Skor bileşenleri

Her seçenek şu sinyallerle puanlanır:

- mevcut Tactical Engine lane score;
- hücum yönünde ilerleme;
- rakip kaleye mesafe kazancı;
- alıcının en yakın savunmacıya mesafesi;
- pas koridoru açıklığı;
- pas mesafesi.

V1 açıklanabilir olmak için rule-based / weighted score kullanır.

## Kritik altyapı koşulu

Önce identity-aware Tactical Engine v1.1 üret:

```powershell
python scripts\build_tactical_v11_identity.py

python -m scripts.tactical_engine_v11_identity `
  --source "input\gsGol1_goal_window.mp4" `
  --team-jsonl "output\gsGol1_goal_team_v29.jsonl" `
  --possession-jsonl "output\gsGol1_goal_possession_v11_v25.jsonl" `
  --output "output\gsGol1_goal_tactical_v29_v11.mp4" `
  --jsonl "output\gsGol1_goal_tactical_v29_v11.jsonl"
```

## Test

```powershell
python -m pytest tests\test_pass_options_ranking_v1.py -v
```

## gsGol1

```powershell
python -m scripts.rank_pass_options_v1 `
  --source "input\gsGol1_goal_window.mp4" `
  --team-jsonl "output\gsGol1_goal_team_v29.jsonl" `
  --possession-jsonl "output\gsGol1_goal_possession_v11_v25.jsonl" `
  --direction-jsonl "output\gsGol1_goal_defline_v11.jsonl" `
  --tactical-jsonl "output\gsGol1_goal_tactical_v29_v11.jsonl" `
  --output "output\gsGol1_goal_pass_options_v1.mp4" `
  --jsonl "output\gsGol1_goal_pass_options_v1.jsonl"
```

## Sonraki aşama

Bu JSONL, Error Detection timeline ile birleştirilecek:

```text
ERR-000x
   ↓
hata anı / peak frame
   ↓
o frame'deki BEST / GOOD pas opsiyonları
   ↓
sonraki possession / shot event
   ↓
Analyst Incident
```

Sonra Analyst Renderer v2:

```text
HATA
 → NEDEN
 → EN İYİ PAS / KOŞU
 → GERÇEKTE NE OLDU?
 → SONUÇ
```
