# Analyst Renderer v2 — Çift Perspektifli Analiz Videosu

Bu renderer artık debug metrik videosu üretmek için değil, gerçek analist anlatısı
oluşturmak için tasarlanmıştır.

## Incident seçimi

Her hata olayı gösterilmez.

Incident'lar şu sinyallerle önceliklendirilir:

- `shot_detected`
- `Attack Merit`
- `Defense Vulnerability`
- birden fazla hata türünün aynı anda görülmesi

Aşırı örtüşen incident'lar bastırılır.

Örneğin mevcut gsGol1 sonucunda:

```text
INC-0002
attack_merit=HIGH
def_vulnerability=HIGH
shot=True
```

en yüksek öncelikli olaydır.

## Video sırası

Her seçilmiş incident:

### A. SAVUNMA BAKIŞI

Peak frame dondurulur.

Gösterilir:
- geç baskı;
- markaj problemi;
- açık pas koridoru;
- savunmanın zafiyeti.

### B. HÜCUM BAKIŞI

Aynı frame farklı anlamla gösterilir.

Gösterilir:
- markajdan kopan oyuncu;
- BEST / GOOD pas;
- hücumun alanı nasıl kullandığı.

### C. ALTERNATİF SAVUNMA

Taktik alternatif metni.

Bu bölüm kesin nedensel hüküm değildir.

### D. GERÇEK AKSİYON

Video yeniden oynar.

Sonuç:
- pas;
- devam eden hücum;
- `SHOT_FLIGHT`;
- ileride goal event.

## Test

```powershell
python -m pytest tests\test_analyst_renderer_v2.py -v
```

## gsGol1

```powershell
python -m scripts.render_analyst_v2 `
  --source "input\gsGol1_goal_window.mp4" `
  --team-jsonl "output\gsGol1_goal_team_v29.jsonl" `
  --errors-timeline-json "output\gsGol1_goal_errors_v11_identity_timeline.json" `
  --marking-jsonl "output\gsGol1_goal_marking_v1.jsonl" `
  --pass-options-jsonl "output\gsGol1_goal_pass_options_v1.jsonl" `
  --shot-jsonl "output\gsGol1_goal_possession_v16.jsonl" `
  --incidents-json "output\gsGol1_goal_analyst_incidents_v1.json" `
  --max-incidents 3 `
  --output "output\gsGol1_goal_analyst_v2.mp4" `
  --storyboard-json "output\gsGol1_goal_analyst_v2_storyboard.json"
```

## Ürün yönü

Bu storyboard yapısı daha sonra iki renderer'a ayrılabilir:

```text
Analyst Incident
   ├── Broadcast / Telestration Renderer
   └── Anime Reconstruction Renderer
```

Anime renderer aynı incident verisini kullanabilir:

```text
SAVUNMA BAKIŞI
→ dramatik freeze frame

HÜCUM BAKIŞI
→ koşu / pas çizgisi

GERÇEK AKSİYON
→ özgün anime yeniden canlandırma
```
