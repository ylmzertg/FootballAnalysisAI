# FootballAnalysisAI — Güncel Durum ve Kaldığımız Yer

**Tarih:** 2026-08-21  
**Proje:** FootballAnalysisAI  
**Amaç:** Yeni bilgisayara taşınabilir futbol analiz sistemi + çoklu dil Analyst/Anime video üretimi.

---

## 1. Şu anki doğrulanmış ana hat

Aşağıdaki parçalar mevcut çalışma hattının doğrulanmış / kullanılabilir tarafıdır:

```text
Video
  ↓
Player Detection + ByteTrack
  ↓
PnLCalib PRIMARY
TVCalib FALLBACK
  ↓
Team Identity V2.9
  ↓
Possession / Direction / Tactical
  ↓
Error Detection V1.1
  ↓
Marking Analysis
  ↓
Pass Options Ranking V1
  ↓
Analyst Incident V1
  ↓
Decision Comparison V1
  ↓
Analyst Renderer V2
```

### Team Identity

**Team Identity V2.9** ilk başarılı deep-embedding baseline olarak kabul edildi.

Önemli sonuç:

```text
Cluster counts      : {1:16, 0:20}
Cluster mapping     : {0:'TEAM_B', 1:'TEAM_A'}
Mapping confidence  : 0.810
Overridden segments : 7
```

Karar:

> V2.9 production baseline olarak donduruldu.

---

## 2. Ball Tracking durumu

### Production

Şimdilik:

```text
Ball Tracker V1
```

production baseline olarak kullanılmaya devam ediyor.

### Deneysel ve dondurulmuş

Aşağıdakiler production'a alınmadı:

```text
Ball v2.0
Ball v2.1
Ball v2.2
Ball v2.2.1
Ball v2.3
```

V2.3 bazı metriklerde iyi görünmesine rağmen boş çim üzerinde false-positive problemi tamamen çözülmedi.

Karar:

> Ball v2.x threshold tuning şimdilik durduruldu.
> Gelecekte independent ball model / optical flow yaklaşımıyla yeniden ele alınacak.

Bu nedenle şu dosyaların `??` olarak kalması bilinçlidir:

```text
core/ball_candidate_verifier.py
core/ball_secondary_redetection.py
core/ball_tracker_v2.py

scripts/build_track_ball_v21.py
scripts/build_track_ball_v22.py
scripts/build_track_ball_v23.py

scripts/track_ball_v2.py
scripts/track_ball_v21.py
scripts/track_ball_v22.py
scripts/track_ball_v23.py

tests/test_ball_candidate_gate_v22.py
tests/test_ball_candidate_verifier.py
tests/test_ball_secondary_redetection.py
tests/test_ball_tracker_v2.py
```

---

## 3. Team Identity deneyleri

Aşağıdaki sürümler başarısız/deneysel kabul edildi:

```text
V2.6 — segment + jersey reconciliation
V2.7 — unsupervised global jersey clustering
V2.8 — two-zone kit signature
```

Production baseline:

```text
V2.9 — deep MobileNetV3 embedding + balanced clustering
```

Bu nedenle aşağıdaki eski deneysel dosyalar production commit'ine alınmak zorunda değildir:

```text
core/team_identity_reconciler_v26.py
core/team_global_clustering_v27.py
core/team_kit_signature_v28.py

scripts/reconcile_teams_v26.py
scripts/reconcile_teams_v27_global_cluster.py
scripts/reconcile_teams_v28_kit_signature.py

tests/test_team_identity_reconciler_v26.py
tests/test_team_global_clustering_v27.py
tests/test_team_kit_signature_v28.py
```

---

## 4. Tactical / Error Detection

Önemli bir hata düzeltildi:

Eski tactical / shape scriptleri doğrudan:

```text
team_v24
```

okuyordu.

Yeni priority:

```text
team_v29
→ team_v28
→ team_v27
→ team_v26
→ team_v25
→ team_v24
```

Identity-aware tactical engine üretildi.

### Error Detection V1.1

Frame-level gürültü temporal event'lere dönüştürüldü.

Gerçek analiz artık:

```text
FREE_PASSING_LANE
LATE_PRESSURE
UNMARKED_RUNNER
```

gibi sinyalleri sequence/event düzeyinde birleştiriyor.

---

## 5. Analyst Incident V1

En önemli mimari karar:

Bir gol veya pozisyon tek bir:

```text
"defans hatası"
```

veya:

```text
"hücum başarısı"
```

etiketine indirgenmeyecek.

Her incident aynı anda şu perspektifleri taşıyabilir:

```text
ATTACK VIEW
DEFENSE VIEW
ALTERNATIVE VIEW
OUTCOME VIEW
```

Örnek:

```text
INC-0002
frames = 84..115
peak = 91
attack = TEAM_B

Attack Merit = HIGH
Defense Vulnerability = HIGH

Errors:
- FREE_PASSING_LANE
- LATE_PRESSURE
- UNMARKED_RUNNER

shot = True
```

Bu incident, mevcut test klibindeki ana gerçek analiz olayıdır.

---

## 6. Marking Analysis

Hücum oyuncuları:

```text
TIGHT
MARKED
LOOSE
UNMARKED
```

olarak değerlendiriliyor.

Ancak:

> "Oyuncu boş = savunma hatası"

şeklinde basit karar verilmiyor.

Tehdit değerlendirmesinde:

```text
topa uzaklık
hücum yönünde ilerleme
kaleye uzaklık
pas opsiyonu
nearest defender distance
threat score
```

birlikte kullanılıyor.

---

## 7. Pass Options Ranking V1

Her confirmed possession frame'inde pas seçenekleri sıralanıyor.

Kategoriler:

```text
BEST
GOOD
RISKY
BLOCKED
```

Önemli terminoloji:

> `BEST` mutlak doğru karar anlamına gelmez.

Doğru ifade:

```text
Sistemin en yüksek puanlı opsiyonu
```

Çünkü mevcut model açıklanabilir weighted/heuristic ranking modelidir.

Henüz learned:

```text
xT
EPV
optimal action
```

modeli değildir.

---

## 8. Decision Comparison V1

Yeni katman:

```text
Sistemin en yüksek puanlı opsiyonu
                VS
Oyuncunun gerçek kararı
```

Karşılaştırma durumları:

```text
MATCHED_BEST
CHOSE_ALTERNATIVE
SHOT_OVER_PASS
NO_CLEAR_COMPARISON
```

Bu katman Analyst Renderer ve ileride Anime Renderer için kullanılacaktır.

Decision Comparison JSON üretimi çalıştırıldı.

---

## 9. Analyst Renderer

### Analyst Renderer V2

Çalışan ilk gerçek anlatı renderer'ı üretildi.

Incident sırası:

```text
SAVUNMA BAKIŞI
      ↓
HÜCUM BAKIŞI
      ↓
ALTERNATİF SAVUNMA
      ↓
GERÇEK AKSİYON
```

`gsGol1_goal_analyst_v2.mp4` üretildi ve incelendi.

Karar:

> Teknik debug videosundan gerçek analyst-storytelling video yapısına geçiş başarılı.

### Analyst Renderer V2.1

Hedef sıra:

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

### ÖNEMLİ: V2.1 builder problemi

Şu dosya:

```text
scripts/build_analyst_renderer_v21.py
```

yerel `render_analyst_v2.py` içinde birebir import string aradığı için:

```text
RuntimeError:
decision helper import:
expected exactly one match, found 0
```

hatası verdi.

Karar:

> Patch/builder yaklaşımı bırakılacak.

Sonraki renderer:

```text
scripts/render_analyst_v21.py
```

**standalone** hazırlanacak.

Bu nedenle şu dosyalar henüz production kabul edilmemeli:

```text
scripts/build_analyst_renderer_v21.py
tests/test_analyst_decision_overlay_v21.py
docs/ANALYST_RENDERER_V21_DECISION.md
```

---

## 10. Çoklu dil kararı

Analiz motoru dile bağlı olmayacak.

Desteklenen ilk diller:

```text
tr — Türkçe
en — English
es — Español
de — Deutsch
fr — Français
pt — Português
```

### Kullanım

Dil parametre ile verilebilir:

```powershell
--language en
```

Bu durumda sistem soru sormaz.

Dil verilmezse video oluşturulmadan önce:

```text
Video dili seçin:

1. Türkçe
2. English
3. Español
4. Deutsch
5. Français
6. Português
```

sorulmalıdır.

### Mimari

```text
Analysis JSON
     ↓
language-neutral semantic data
     ↓
video_i18n
     ↓
TR / EN / ES / DE / FR / PT
```

Aynı analiz tekrar hesaplanmadan farklı dillerde yeniden render edilebilir.

Ortak modül:

```text
core/video_i18n.py
```

Bu language code ileride aynı zamanda:

```text
overlay
narration script
TTS voice
subtitle
YouTube metadata
```

katmanlarını kontrol edebilir.

---

## 11. Anime / yeniden canlandırma projesi

Analiz motoru tekrar yazılmayacak.

Aynı semantic incident verisi:

```text
Analysis Engine
   ├── Broadcast / Telestration Renderer
   └── Anime Reconstruction Renderer
```

şeklinde iki farklı görsel renderer'a gidecek.

Anime yaklaşımı:

- dramatik freeze
- koşu vurgusu
- speed lines
- impact frame
- uzatılmış koşu/şut anlatısı
- futbol anime mantığı

Ancak mevcut anime karakterlerini veya özgün eser görsellerini kopyalamadan, özgün stil kullanılacak.

---

# 12. Yeni bilgisayara taşınabilirlik

Mevcut repo içinde zaten:

```text
setup_windows.ps1
scripts/health_check.py
scripts/install_engines.ps1
scripts/download_models.ps1
requirements/main-windows-cpu.txt
```

bulunuyor.

Mevcut setup:

```text
Python 3.10
main .venv
PyTorch CPU
main requirements
Python 3.9
PnLCalib
TVCalib
model downloads
local.yaml
health check
```

akışını kuruyor.

## Yeni bilgisayarda hedef kullanım

```powershell
git clone https://github.com/ylmzertg/FootballAnalysisAI.git
cd FootballAnalysisAI

.\setup_windows.ps1
.\.venv\Scripts\Activate.ps1

python scripts\health_check.py
python scripts\portable_acceptance.py
```

---

# 13. ŞU ANDA TAM OLARAK KALDIĞIMIZ YER

## Tamamlanan son iş

Production / doğrulanmış dosyalar Git staging için seçildi.

Deneysel dosyalar bilinçli şekilde `??` olarak bırakıldı.

Özellikle aşağıdaki grupların `??` kalması normal:

```text
Ball Tracker v2.x
Team Identity v2.6
Team Identity v2.7
Team Identity v2.8
Analyst Renderer v2.1 broken builder
```

## Bir sonraki geliştirme adımı

### ÖNCE: Portable Setup V2

Analyst Renderer'a devam etmeden önce yeni bilgisayara geçişi sağlamlaştır.

Yapılacaklar:

1. `scripts/portable_acceptance.py`
2. health-check'e football ball model kontrolü
3. critical V2.9 / Shot V1.6 / i18n regression kontrolleri
4. `docs/NEW_PC_SETUP.md`
5. model/assets manifest
6. CPU baseline korunacak
7. ileride CUDA profile desteği

Hedef:

```text
FootballAnalysisAI

Python              OK
Player Model         OK
Ball Model           OK
PnLCalib             OK
TVCalib              OK
Team Identity V2.9   OK
Shot V1.6            OK
i18n                 OK
Pipeline              OK

READY FOR DEVELOPMENT
```

### SONRA: Analyst Renderer V2.1 Standalone

Portable Setup V2 doğrulandıktan sonra:

```text
standalone render_analyst_v21.py
+ Decision Comparison
+ interactive language selection
+ --language parameter
```

hazırlanacak.

Ardından:

```text
Analyst Renderer V2.2
```

için:

- TV-style telestration
- freeze/zoom
- markaj bağlantı çizgileri
- savunma hattı
- gerçek aksiyon path
- daha kısa profesyonel metinler
- incident seçimi / anlatı sadeleştirmesi

çalışılacak.

---

# 14. Git çalışma kuralı

Bundan sonra:

> `git add -A` kullanılmayacak.

Milestone bazlı seçilmiş dosyalar commit edilecek.

Deneysel kod:

```text
production milestone
```

ile karıştırılmayacak.

Özellikle Ball v2.x ve eski Team Identity deneyleri ayrı tutulacak.

---

# 15. Referans test klibi

Ana regression klibi:

```text
input/gsGol1_goal_window.mp4
```

Önemli golden beklentiler:

```text
TEAM_A attack direction = PLUS_X
TEAM_B attack direction = MINUS_X

Shot:
~ frames 127..132
TEAM_B
SHOT_FLIGHT

Analyst Incident:
INC-0002
Attack Merit = HIGH
Defense Vulnerability = HIGH
shot = True
```

Yeni bilgisayara geçildiğinde bu golden değerler portable acceptance test için kullanılabilir.
