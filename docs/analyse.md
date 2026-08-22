# FootballAnalysisAI - Analyse

**Last Updated:** 2026-08-22  
**Repository:** FootballAnalysisAI  
**Purpose:** Bu doküman proje içindeki mevcut analizleri, tamamlanan üretim bileşenlerini, planlanan geliştirmeleri, faydalı olabilecek ek fikirleri, referans içerikleri ve son teknik durumları tek yerde toplar.

---

# 1. Project Goal

FootballAnalysisAI amacı, gerçek maç videosu üzerinden:

- saha kalibrasyonu
- oyuncu / top tespiti ve takibi
- takım kimliği çözümü
- topa sahip olma analizi
- hücum yönü / savunma hattı analizi
- taktiksel bağlam üretimi
- hata tespiti
- alternatif pas opsiyonları
- markaj analizi
- analist olay akışı
- karar karşılaştırması
- profesyonel analiz videosu üretimi

yapabilen bir sistem oluşturmaktır.

İkinci aşamada ise bu semantik analizlerden yararlanarak sahneleri anime / reconstruction formatına dönüştürmek hedeflenmektedir. Bu ikinci aşama ayrı repo olarak ele alınacaktır.

---

# 2. Production Baseline (Current)

## Active Production Stack

- PnLCalib PRIMARY
- TVCalib FALLBACK
- Team Identity V2.9
- Ball Tracker V1
- Possession V11 / V12 Events
- Direction V11
- Tactical V11
- Shot Context V16
- Error Detection V1.1 / V11
- Pass Options Ranking V1
- Marking Analysis V1
- Analyst Incident V1
- Decision Comparison V1
- Analyst Renderer V2 / V2.1

## Non-Production / Experimental

- Ball v2.x experimental
- Team Identity v2.6-v2.8 experimental
- Broken Analyst Renderer V2.1 builder from older branch is not baseline

---

# 3. Completed / Existing Analysis Modules

## 3.1 Computer Vision Base
- Player detection
- Pitch detection
- Ball detection
- ByteTrack based tracking
- Wide shot / scene support
- PnLCalib integration
- TVCalib integration
- Calibration fusion preparation

## 3.2 Team / Ball / Possession
- Team classification V2.x progression
- Team Identity V2.9
- Ball Tracker V1
- Possession estimation V11
- Possession events V12 state machine

## 3.3 Tactical / Contextual Analysis
- Direction V11
- Tactical V11
- Shot Context V16
- Error Detection V11
- Pass Options Ranking V1
- Marking Analysis V1
- Analyst Incident Builder V1
- Decision Comparison V1

## 3.4 Rendering
- Analyst Renderer V2
- Analyst Renderer V2.1
- Multilingual label preparation for analyst rendering
- Production Pipeline V2 runner

---

# 4. Portable Setup / New PC Status

## Validated on New Windows PC
- Repo cloned successfully
- Portable Setup V2 installed
- Python 3.10 virtual environment installed
- CUDA-enabled main environment active
- PnLCalib installed
- TVCalib installed with Windows compatibility patches
- Ball / Player / Pitch models downloaded and verified
- Health Check V2 passed
- Portable Acceptance passed

## Important portability fixes already required
- Missing `core.deep_kit_encoder_v29.py` was restored
- Missing `core.team_identity_reconciler_v26.py` was restored
- TVCalib Windows compatibility patches applied
- Model manifest / download verification support added

---

# 5. Current Production Pipeline V2 Flow

Current intended sequence:

1. tracking / base detections  
2. calibration  
3. team_v29  
4. ball_v1  
5. possession_v11  
6. possession_events_v12  
7. direction_v11  
8. tactical_v11  
9. shot_v16  
10. errors_v11  
11. pass_options_v1  
12. analyst_incidents_v1  
13. decision_comparison_v1  
14. analyst_renderer_v21  

---

# 6. Latest Real Pipeline Run (osimhen_10s)

## Input
- `input/osimhen_10s.mp4`

## Run Name
- `osimhen_prod_v2`

## Observed Progress
- PnLCalib accepted 18/18 samples
- Team Identity V2.9 step reached and continued after restoring missing runtime files
- Possession Events V1.2 completed successfully

## Possession Events V1.2 Result
- Frames processed: 250
- TEAM_A team-state: 15
- TEAM_B team-state: 90
- LOOSE: 144
- UNKNOWN: 1

### Phase Breakdown
- CONTROL: 46
- CONTROL_GAP: 1
- PASS_FLIGHT: 58
- TEAM_FLIGHT: 0
- CONTESTED_FLIGHT: 72
- RAW_LOOSE: 72
- RAW_UNKNOWN: 1

### Generated Outputs
- `output/production_v2/osimhen_prod_v2/06_possession_events_v12.mp4`
- `output/production_v2/osimhen_prod_v2/06_possession_events_v12.jsonl`

## Latest Failure
Pipeline failed at:

- `scripts.attack_direction_defline_v11`

### Error
- `ModuleNotFoundError: No module named 'core.attack_direction'`

### Meaning
- Direction V11 step cannot start because `core.attack_direction` runtime dependency is missing or not correctly committed.

---

# 7. Planned Development (Priority)

## 7.1 Immediate
- Fix `core.attack_direction` missing dependency
- Resume Production Pipeline V2 from `direction_v11`
- Complete real `14_analyst_v21.mp4` output generation
- Validate analyst video output quality

## 7.2 Near-Term
- Standalone Analyst Renderer V2.1 stabilization
- Decision Comparison integration validation
- Multilingual runtime support
- `--language` parameter support
- TR / EN / ES / DE / FR / PT

## 7.3 Analyst Renderer V2.2
- TV-style telestration
- Freeze / zoom moments
- Marking lines
- Defensive line overlay
- Actual action path visualization
- Professional concise narration feel

---

# 8. Analysis Ideas We Definitely Want

## 8.1 Player Name Overlay
Oyuncuların isimlerini sürekli değil, kritik anlarda göstermek:
- top sahibi olduğunda
- markaj analizi sırasında
- karar anında
- önemli koşu / pas alıcısı olduğunda

Bu özellik için gelecekte:
- jersey number recognition
- roster mapping
- player identity overlay

katmanları gerekecektir.

## 8.2 Run Opportunity / Space Analysis
Yalnızca pas değil, koşu koridorlarını da göstermek:
- potansiyel koşu alanı
- savunma hattı arkası boşluk
- rakipler arası koridor
- ofsayt çizgisine göre değerlendirilen koşu seçeneği

Bu, sistemin “yüksek değerli koşu alternatifi” üretmesini sağlayacaktır.

## 8.3 More Detailed Tactical Narration
- neden hata olduğu
- neden bu pasın daha iyi olduğu
- neden bu koşunun alan yaratacağı
- savunmanın hangi oyuncusu geç tepki verdi
- karar anı ile gerçek aksiyonun farkı

---

# 9. Nice-to-Have / Future Good Ideas

- Player role hints
- Pressing traps
- Space occupation heat moments
- Defensive compactness
- Passing lane closure timing
- Recovery run highlighting
- Multi-camera support
- More automated incident selection
- More cinematic analyst rendering
- Highlight packaging for YouTube shorts

---

# 10. Separate Anime / Reconstruction Track

Anime / reconstruction aşaması aynı repo içinde tutulmamalıdır.

## Recommended Architecture

### Repo 1
- `FootballAnalysisAI`
- gerçek video analizi
- incident JSON üretimi
- telestration / analyst rendering

### Repo 2
- `FootballAnimeAI` veya `FootballReconstructionAI`
- semantik incident verisini okuyarak
- 2D / 3D / anime tarzı
- yeniden canlandırma
- farklı kamera / anlatım / efekt

## Shared Contract
İki repo arasında ortak format:
- incident JSON
- player positions
- ball path
- decision frame
- tactical alternatives
- annotation metadata

---

# 11. Reference Content

## X / Twitter examples
- `https://x.com/ekgedik1903/status/2088254077234823609/video/1?s=46`
- `https://x.com/alex10tr10/status/2090951425056448812/video/1?s=48`

Bu örnekler orijinal video üzerinde analiz-telestration yönü için değerlidir.

## YouTube reference
- `https://www.youtube.com/watch?v=neBZ6huolkg&list=PLbpuDsjUU7EuxKRav7eSoRw3M4fSW-ii2`

Bu içerikler de analiz mantığı, anlatım akışı ve sahneleme açısından referans kabul edilir.

## GitHub reference
- `https://github.com/abdullahtarek/football_analysis`

Bu repo referans amaçlı incelenmiştir; fikir almak için yararlıdır ancak birebir baseline değildir.

---

# 12. Important Project Decisions

- Portable Windows Setup V2 korunacak
- Production baseline korunacak
- Experimental branch mantığıyla ilerlenmeli
- Missing runtime dependencies tek tek portable hale getirilmeli
- Önce gerçek video analiz motoru stabilize edilmeli
- Anime / reconstruction ayrı repo olmalı
- Analiz kararları ve fikirleri bu dosyada merkezi olarak tutulmalı

---

# 13. Open Issues

## Runtime / Missing Modules
- `core.attack_direction` missing in current runtime

## Pipeline Continuation Need
- resume from `direction_v11`

## Documentation
- other scattered notes can later be merged into this central file
- this file should become the single source of truth for analysis scope

---

# 14. Next Action

1. Fix `core.attack_direction`
2. Resume pipeline from `direction_v11`
3. Finish analyst output
4. Review generated analyst video
5. Commit and push:
   - `docs/analyse.md`
   - code fixes for missing runtime modules
   - any pipeline stabilization changes

---

## 15. Real Video Quality Findings - Osimhen / Roboflow

**Date:** 2026-08-22

### 15.1 Analyst Renderer V2.1 is technically working but not yet production-quality

The first end-to-end Production Pipeline V2 run successfully produced:

- `14_analyst_v21.mp4`
- `14_analyst_v21_storyboard.json`
- `manifest.json`

However, visual inspection showed football-semantic errors.

Important decision:

> A pipeline finishing successfully is not enough. Tactical and football-semantic correctness must be validated before an analyst video is accepted.

### 15.2 Team Identity V2.9 can be internally consistent but visually wrong

Osimhen test diagnostics:

- cluster counts: 18 / 18
- mapping confidence: approximately 0.72
- embedding device: CUDA
- marking same-team integrity violations: 0

Despite those internally consistent numbers, visual inspection showed some players from visibly different kit groups being assigned to the same team.

Example pattern:

- possessor classified as TEAM_B
- BEST pass receiver also classified as TEAM_B
- visually, the two players appear to belong to different teams

Consequence:

`Team Identity error -> Pass Options trusts wrong team -> Renderer draws wrong tactical arrow`

Therefore internal TEAM_A / TEAM_B consistency checks alone are insufficient.

### 15.3 Tactical Integrity Gate is mandatory

Before tactical data reaches the renderer, a new validation layer should reject or suppress impossible relationships.

Required checks:

- suggested pass receiver must belong to the possessor's team
- actual completed pass receiver must belong to the source player's team
- marking attacker and defender must belong to opposite teams
- defensive line members must belong to the same defending team
- TEAM_A and TEAM_B must not resolve to the same attacking direction
- low-confidence identity must not generate strong tactical claims
- suspicious team assignments should be checked against independent visual kit evidence
- if confidence is insufficient, prefer UNKNOWN / uncertain instead of drawing a misleading arrow

Planned component:

`Tactical Integrity Gate V1`

### 15.4 Decision Comparison turnover bug

Osimhen Decision Comparison showed cases such as:

- possessor = TEAM_A
- actual receiver = TEAM_B

and:

- possessor = TEAM_B
- actual receiver = TEAM_A

These should not automatically be interpreted as successful passes.

Possible real meanings:

- interception
- turnover
- deflection
- contested ball
- opponent recovery

Required change:

`Decision Comparison V1.1`

A PASS should only be considered a completed same-team pass when source and target identities satisfy the team-consistency guard.

Otherwise the result should be classified as turnover / contested / unresolved rather than normal PASS.

### 15.5 Team Identity V3 direction

Team Identity V2.9 remains the current baseline, but the next identity architecture should improve visual robustness.

Proposed Team Identity V3:

Player detection
-> central torso / jersey crop
-> visual embedding
-> colour descriptor
-> feature fusion
-> clustering
-> temporal segment voting
-> appearance sanity check
-> TEAM_A / TEAM_B / UNKNOWN

Important design change:

Do not rely primarily on the full player bounding box.

Use a central torso / jersey crop to reduce contamination from:

- grass
- opponent players
- overlapping bodies
- crowd/background
- shorts/socks when they have misleading colours

### 15.6 Roboflow technical reference

Reference channel:

`https://www.youtube.com/@Roboflow`

Reference video:

`https://www.youtube.com/watch?v=ukQkeqE0RUI`

Roboflow sports-computer-vision work should be treated as an important technical reference, especially for:

- player detection and tracking
- ball tracking
- sports-specific computer vision
- team classification
- player re-identification
- jersey-number recognition
- camera calibration
- pitch/radar visualisation

Useful ideas to evaluate:

- central player / jersey crops
- modern visual embeddings such as SigLIP-style features
- embedding clustering
- temporal identity stabilisation
- jersey number recognition
- player re-identification

These ideas are references, not automatic replacements for the existing production stack.

### 15.7 What we keep from the current architecture

Do not regress to simple approaches just because reference projects use them.

Keep:

- PnLCalib as PRIMARY calibration
- TVCalib as FALLBACK
- Team Identity temporal logic
- Ball Tracker V1 as current production baseline
- Possession event state machine
- Direction / Defensive Line
- Tactical Engine
- Error Detection
- Pass Options
- Marking
- Analyst Incidents
- Decision Comparison
- Analyst Renderer

Improve weak layers individually.

### 15.8 Player Identity / Name Overlay

Planned future feature:

Player Detection
-> Track ID
-> Jersey Number Recognition
-> Team Roster Mapping
-> Player Name

Player names should not remain on screen continuously.

Use them selectively:

- first freeze frame
- important ball possession
- key run
- marking event
- decision moment
- actual vs alternative action

This should help viewers understand the analysis without making the video look like a debug interface.

### 15.9 Run Opportunity / Space Engine

Detailed analysis should include more than pass alternatives.

Planned:

`Run Opportunity / Space Engine V1`

Potential signals:

- defensive line position
- open channels
- opponent occupancy
- distance to goal
- forward progression
- passing accessibility
- offside line
- expected defender closing time
- current player movement
- whether the space remains open over subsequent frames

Output terminology should avoid claiming certainty.

Preferred wording:

`High-value run opportunity detected by the system`

rather than:

`This was definitely the correct run`

### 15.10 Product differentiation

External reference projects are strongest in:

- detection
- tracking
- calibration
- coordinate extraction
- sports CV

FootballAnalysisAI should differentiate itself through:

- tactical intelligence
- marking analysis
- pass alternatives
- run alternatives
- defensive errors
- decision comparison
- tactical integrity validation
- professional telestration
- analyst storytelling

### 15.11 Immediate priority after the Osimhen test

Development priority:

1. Fix Decision Comparison same-team / turnover logic
2. Improve Team Identity visual reliability
3. Add Tactical Integrity Gate
4. Validate Direction / Defensive Line sanity
5. Re-run semantic outputs
6. Only then improve Analyst Renderer V2.2 visuals

Important rule:

> Do not polish renderer graphics while the underlying football semantics are still unreliable.

## Milestone 2026-08-22 - Team Identity V3.1 + Tactical Integrity Gate V1

### Decision Comparison V1.1

- `ACTUAL_TURNOVER` desteği eklendi.
- `TURNOVER_OVER_PASS` comparison sonucu eklendi.
- Same-team / turnover guard aktif:
  - aynı takım -> `PASS`
  - rakip takım -> `TURNOVER`
  - belirsiz takım -> `UNKNOWN`
- Decision Comparison testleri geçti: `5 passed`.
- Renderer + Decision Comparison birlikte geçti: `8 passed`.

### Analyst Renderer V2.1

`TURNOVER` ve `TURNOVER_OVER_PASS` sonuçlarının renderer tarafından desteklenmesi sağlandı.

Çoklu dil desteğine şu anahtarlar eklendi:

- `actual_turnover`
- `turnover_over_pass`

Desteklenen diller:

- TR
- EN
- ES
- DE
- FR
- PT

### Team Identity V3 deneyi

İlk V3 yaklaşımında deep embedding için yalnızca üst gövde / jersey crop kullanıldı.

Sonuç:
- Takım ayrımını yeterince iyileştirmedi.
- Bu yaklaşım tek başına yeterli görülmedi.
- V3 adayı referans/deney olarak bırakıldı.

### Team Identity V3.1

Yeni yaklaşım:

- jersey-focused crop
- explicit jersey colour descriptor
- HSV + Lab renk histogramları
- deep embedding + jersey colour fusion

Ağırlıklar:

- Deep embedding: `0.35`
- Jersey colour: `0.65`

Osimhen 10 saniye ürün testi sonucu belirgin şekilde iyileşti.

Kritik doğrulamalar:

- Frame 23:
  - possessor = ID 20 / TEAM_B
  - önceki hatalı BEST = ID 8 artık pas seçenekleri içinde değil.
  - adaylar ID 16, 26, 18.
- Frame 91:
  - yanlış BEST üretilmiyor.
  - yalnızca RISKY seçenekler mevcut.
- Frame 150:
  - ID 9 = BEST
  - ID 4 = GOOD
  - ID 14 = RISKY
- Frame 162:
  - güvenilir pas seçeneği yoksa sistem seçenek üretmiyor.

23 / 91 / 150 / 162 kontrolünde tüm üretilen pass option alıcılarında:

`SAME_TEAM = True`

Bu nedenle Team Identity V3.1 mevcut Osimhen milestone'u için başarılı kabul edildi.

### Tactical Integrity Gate V1

Yeni dosya:

`scripts/apply_tactical_integrity_gate_v1.py`

Gate şu kontrolleri yapıyor:

1. Rakip takım receiver -> DROP
2. Çok yakın baskıdaki receiver -> BEST olamaz
3. Dar passing lane -> BEST olamaz
4. BEST ile ikinci seçenek arasında yeterli skor farkı yoksa BEST aşağı çekilir
5. BLOCKED seçenek BEST/GOOD'a yükseltilmez
6. Maksimum bir BEST korunur

Osimhen V3.1 sonucu:

- Frames: 250
- Options in: 138
- Options out: 138
- Team mismatch dropped: 0
- BEST demoted by margin: 2
- BEST demoted by space: 0
- BEST demoted by clearance: 0
- BEST kept: 3

Önemli doğrulama:

Frame 150:

- ID 9 BEST = 0.80363
- ID 4 GOOD = 0.74152
- ID 14 RISKY = 0.23159

Tactical Integrity Gate bu BEST kararını korudu.

### Mevcut ürün durumu

Production Pipeline V2 uçtan uca çalışıyor.

Osimhen testinde tespit edilen en büyük yanlış BEST problemi Team Identity V3.1 ile önemli ölçüde azaltıldı.

V3.1 ve Tactical Integrity Gate henüz Production Pipeline V2'nin kalıcı varsayılan adımları haline getirilmedi.

### Moladan sonra

Bir sonraki ana çalışma:

1. Team Identity V3.1'i Production Pipeline V2'ye entegre et
2. Tactical Integrity Gate V1'i pipeline'a entegre et
3. Yeni Osimhen production videosu üret
4. Direction sanity check geliştir
5. Renderer V2.2

Çalışma prensibi:

`Kod -> video üret -> gerçek görüntüyü kontrol et -> yalnızca üründe görülen ana hatayı düzelt`

Küçük JSON / diagnostik ayrıntılarda gereksiz yere dolaşılmayacak.
