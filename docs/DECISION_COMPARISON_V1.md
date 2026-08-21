# Decision Comparison v1

Amaç:

```text
Sistem hangi pası en iyi seçenek olarak görüyordu?
                VS
Oyuncu gerçekte ne yaptı?
```

Bu katman Analyst Renderer ve Anime Renderer için çok değerlidir.

Örnek:

```text
BEST option: ID 18, score 0.81
Actual action: PASS -> ID 7

Analyst note:
"Sistem merkezdeki ID18'i daha güçlü progresif seçenek olarak görürken
top sahibi sağdaki ID7'yi tercih etti."
```

veya:

```text
BEST option: ID 18
Actual action: SHOT

"Alternatif pas seçeneği vardı ancak oyuncu şutu tercih etti."
```

## Neden önemli?

Böylece analiz yalnız savunma hatası tespiti değildir.

Aynı pozisyonda:

- savunmanın neyi yanlış yaptığı;
- hücumun hangi alanı yarattığı;
- oyuncunun hangi seçeneklere sahip olduğu;
- gerçekte hangi kararı verdiği

birlikte gösterilebilir.

## Test

```powershell
python -m pytest tests\test_decision_comparison_v1.py -v
```

## gsGol1

```powershell
python -m scripts.compare_decisions_v1 `
  --incidents-json "output\gsGol1_goal_analyst_incidents_v1.json" `
  --pass-options-jsonl "output\gsGol1_goal_pass_options_v1.jsonl" `
  --possession-events-jsonl "output\gsGol1_goal_possession_v12_longflight.jsonl" `
  --shot-jsonl "output\gsGol1_goal_possession_v16.jsonl" `
  --lookahead-frames 45 `
  --output-json "output\gsGol1_goal_decision_comparison_v1.json"
```

Sonraki adım:
Analyst Renderer v2.1 bu veriyi ekranda:

```text
RECOMMENDED
vs
ACTUAL
```

olarak gösterecek.
