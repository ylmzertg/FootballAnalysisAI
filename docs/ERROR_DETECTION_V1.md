# Error Detection v1

İlk açıklanabilir taktik hata katmanı.

Amaçlanan anlatım:

```text
HATA
  -> TEHDİT
  -> PAS / KOŞU OPSİYONU
  -> ATAĞIN DEVAMI
  -> ŞUT / SONUÇ
```

V1 üç aday hata üretir:

- `LATE_PRESSURE`
- `UNMARKED_RUNNER`
- `FREE_PASSING_LANE`

Bunlar kesin futbol hükümleri değil; analist tarafından doğrulanabilecek, açıklanabilir aday olaylardır.

## Test

```powershell
python -m pytest tests\test_error_detection_v1.py -v
```

## gsGol1

```powershell
python -m scripts.detect_errors_v1 `
  --source "input\gsGol1_goal_window.mp4" `
  --team-jsonl "output\gsGol1_goal_team_v29.jsonl" `
  --possession-jsonl "output\gsGol1_goal_possession_v11_v25.jsonl" `
  --direction-jsonl "output\gsGol1_goal_defline_v11.jsonl" `
  --tactical-jsonl "output\pipeline_v1\gsGol1_goal\07_tactical_v11.jsonl" `
  --output "output\gsGol1_goal_errors_v1.mp4" `
  --jsonl "output\gsGol1_goal_errors_v1.jsonl"
```

İlk kalite kapısı:
- işaretlenen hata görüntüde makul olmalı;
- her kareyi hata ilan etmemeli;
- referee/kaleci sıradan savunmacı gibi kullanılmamalı;
- tekrar eden kareler sonraki Event Sequencer sürümünde tek olaya indirgenecek.
