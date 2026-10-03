# MVP Demo Planı — 10 Gün / Part-Time

## 0. Bu belge neden ayrı

Ana proje planı (`proje_plani.md`) **üretim ölçeğini** hedefliyor: 130 kamera,
DeepStream, Kubernetes, LiDAR/mimari plan kalibrasyonu, ROS2. Bu belge ise
**tek amaca** hizmet ediyor: 10 gün içinde, part-time çalışarak, **canlı
sunumda çalışan** bir kanıt-niteliğinde (proof-of-concept) demo çıkarmak.
İkisi çelişmiyor — demo, ana mimarinin "küçültülmüş, hazır kütüphanelerle
kurulu" bir kesiti. §7'de hangi demo bileşeninin ana mimariye nasıl
büyüyeceği gösteriliyor.

**Varsayım:** 4 kişi, günde ortalama 2-3 saat (part-time). Bu, toplamda
~80-120 kişi-saat demek — gerçekçi bir MVP için yeterli, ama sıfır tolerans
var demektir: kapsam disiplinli tutulmalı.

---

## 1. Kapsam Daraltma — Ana Plandan Ne Çıkarıldı, Ne Basitleştirildi

| Ana plandaki bileşen | Demo'da ne oluyor | Neden |
|---|---|---|
| DeepStream + özel YOLO26 bbox-parser | **`ultralytics` Python paketi** (`pip install ultralytics`), hazır `model.track()` fonksiyonu | DeepStream kurulumu (driver, TensorRT, custom parser) günler alır; ultralytics 10 dakikada çalışır, YOLO26 zaten paket içinde hazır |
| ByteTrack'i elle entegre etme | Ultralytics'in **yerleşik** ByteTrack entegrasyonu (`tracker="bytetrack.yaml"`) | Sıfır ek kod, tek parametre |
| PeopleNet/RT-DETR (ayrı insan dedektörü) | **Tek model** (YOLO26, COCO sınıflarıyla: person, chair, couch, dining table, tv, laptop, vb.) | Demo'da iki ayrı model bakımı gereksiz karmaşıklık |
| AMC / `direct_visual_lidar_calibration` / PnP | **Basit homografi** (`cv2.getPerspectiveTransform`, 4 nokta tıklama) | Tek düz zeminli bir oda için homografi matematiksel olarak yeterli, kalibrasyon dakikalar sürer |
| LiDAR SLAM / mimari plan (CAD) | Basit, elle çizilmiş/hazırlanmış **2D oda krokisi** (herhangi bir çizim aracında, ölçekli) | Demo odası küçük ve sabit, karmaşık geometri gerekmiyor |
| Kafka + Mosquitto + ROS2 | **Tek bir WebSocket bağlantısı** (FastAPI üzerinden) | Tek makine, tek işlem — mesajlaşma altyapısı gereksiz |
| Docker Compose (çok container) | **Doğrudan Python/Node çalıştırma** (opsiyonel: tek basit `docker-compose.yml`, 2 servis) | Kurulum hızını maksimize etmek için container'sız da çalışabilir |
| Kubernetes | Yok | Tek makine demo'da anlamsız |
| PostgreSQL / kalıcılık | **Bellek-içi (in-memory) durum** | Demo süresi boyunca (dakikalar) kalıcılık gereksiz |
| Flutter mobil uygulama | Yok (gelecek işi olarak sunumda bahsedilir) | 10 günde web + mobil ikisi birden sığmaz |
| **React frontend'i sıfırdan kurma** | **Yok — frontend zaten hazır.** Bu ekibin işi, frontend'in tüketeceği API'yi **tanımlamak ve kurmak** (şema biz tarafımızdan belirlenip frontend'e iletiliyor) | Kapsam değişikliği (bkz. §1.1) — 10 günlük planın en büyük kalem tasarrufu |
| Çoklu kat / çoklu bölge | **Tek oda, tek kat** | Demo alanı fiziksel olarak sınırlı |
| Çoklu güvenlik kamerası (130) | **1 kamera zorunlu, 2. kamera "stretch goal"** (bkz. §5, Gün 9) | Tek kamerayla bile ana konsept (canlı tespit + harita) tam gösterilir |

**Kritik ilke:** Her basitleştirme, ana plandaki bir bileşenin **yerini
tutuyor**, farklı bir şey icat etmiyor. Yani demo kodu tamamen çöpe gitmiyor
— §7'de bu köprü gösteriliyor.

### 1.1. Kapsam güncellemesi: Frontend zaten hazır, API sözleşmesi bizim tarafımızdan belirlendi

Bu, planı önemli ölçüde değiştiren bir bilgi. Artık iş **"canlı çalışan bir
arka plan sistemi kurup, kendi tanımladığımız API sözleşmesini frontend'e
iletmek."** İki doğrudan sonucu var:

1. **Gün planındaki tüm "React'ı sıfırdan kurma" işleri düşüyor** — toplam
   iş yükünün kabaca %25-30'u kadar bir tasarruf demek.
2. **Frontend'in mevcut kodunu "reverse-engineer" etmeye gerek yok** —
   sözleşme zaten bizim tarafımızdan tanımlandı ve `API_SOZLESMESI.md`
   dosyasında yazılı. Bu, önceki taslakta planlanan "D, frontend kodunu
   inceleyip şemayı çıkarır" adımını gereksiz kılıyor; yerine geçen adım:
   **bu dokümanı Gün 1'de frontend ekibine iletmek.**

**Kararlaştırılan API sözleşmesi** (`API_SOZLESMESI.md`, proje deposunda):

| Uç nokta | Tip | Amaç |
|---|---|---|
| `WS /ws/tracking` | WebSocket | Canlı tracking verisi, ~10-15 Hz, `{type, timestamp, objects: [{id, class, x, y, confidence, is_static}]}` |
| `GET /api/map/config` | REST | Sayfa yüklenince bir kez — harita görseli yolu, gerçek boyut (metre), sınıf-renk şeması |
| `GET /api/objects/latest` | REST | İlk render / WebSocket bağlanmadan önceki anlık durum |
| `GET /api/health` | REST | Sağlık kontrolü |

Hazır bir **FastAPI iskeleti** (`backend/main.py`) ve bağımlılık listesi
(`requirements.txt`) da bu sözleşmeye uygun şekilde hazırlandı — A'nın
tespit/takip kodu bu iskeletin `detection_loop()` fonksiyonuna entegre
edilecek.

**Frontend ekibi için avantaj:** Sözleşme Gün 1'de netleştiği için, onlar
kendi tarafında bu şemaya uygun **sahte (mock) veriyle** hemen geliştirmeye
başlayabilir — bizim gerçek veriyi üretmemizi (Gün 5) beklemek zorunda
kalmazlar. Bu, iki ekibin paralel ilerlemesini sağlıyor.

---

## 2. Kullanılacak Hazır Kütüphaneler

| Katman | Kütüphane | Neden bu seçildi |
|---|---|---|
| Tespit + takip | `ultralytics` (YOLO26n/YOLO26s + yerleşik ByteTrack) | Tek satırda kurulum, tek fonksiyonla (`.track()`) hem tespit hem takip |
| Kamera/görüntü işleme | `opencv-python` | Homografi, kamera yakalama, çizim — endüstri standardı |
| Backend | `fastapi` + `uvicorn` (WebSocket yerleşik) | Asenkron, WebSocket desteği yerleşik, minimal kod |
| Frontend | **Zaten hazır** (dışarıdan verilen) | Bu ekibin kapsamı değil — sadece API sözleşmesine uyum sağlanacak |

**API sözleşmesi (frontend ekibine verilecek):** `WS /ws/tracking`,
`GET /api/map/config`, `GET /api/objects/latest`, `GET /api/health`.
Tam şema ve örnek payload'lar için bkz. proje deposundaki
`API_SOZLESMESI.md` — bu belge frontend ekibiyle Gün 1'de paylaşılmalı.

---

## 3. Ekip Görev Dağılımı (Demo için basitleştirilmiş)

Ana plandaki 4 rol (Perception/Altyapı/Backend/Frontend) aynı kalıyor, ama
demo kapsamında herkesin işi çok daha dar ve paralel:

| Kişi | Demo'daki görevi |
|---|---|
| A (Perception) | YOLO26 + ByteTrack kurulumu, homografi kalibrasyonu, foot-point izdüşüm kodu |
| B (Altyapı) | Geliştirme ortamı kurulumu, demo günü donanım/ağ testi, backup video kaydı, canlı demo prova koordinasyonu |
| C (Backend) | FastAPI + WebSocket/REST servisi, A'nın çıktısını frontend'in beklediği sözleşmeye uygun JSON olarak yayınlama |
| D (**Entegrasyon & Test**, eski "Frontend" rolünün yerine) | `API_SOZLESMESI.md`'yi Gün 1'de frontend ekibine iletir, onların soru/geri bildirimlerini toplar; C ile birlikte uçtan-uca entegrasyon testini yürütür; ikinci kamera füzyonunda (Gün 7-8) A'ya destek olur |

**Faz 0'daki gibi tam bir eğitim turuna 10 günde yer yok** — ama Gün 1'de
30 dakikalık bir **ortak kickoff** (bu belgenin baştan sona okunması + roller
netleştirme) şart, aksi halde entegrasyon günlerinde (Gün 6-7) sürpriz çıkar.

---

## 4. Gün Gün Plan (Güncellendi — Frontend Hazır Varsayımıyla)

Frontend zaten var olduğu için önceki plandaki "React'ı sıfırdan kurma"
günleri tamamen kalktı; bu, ~2-3 günlük bir zaman kazancı demek. Bu kazanılan
zaman, §4'ün sonunda 2. kamera füzyonunu (eski "stretch goal") **daha
güvenceli bir hedefe** çeviriyor ve sağlamlaştırmaya daha fazla gün ayırıyor.

### Gün 1 — Ortak kickoff + ortam kurulumu + sözleşme teslimi
- **Herkes (30 dk):** Bu planın okunması, rollerin netleştirilmesi, demo
  senaryosunun (bkz. §5) üzerinden geçilmesi.
- **A:** `pip install ultralytics opencv-python`, `yolo26n.pt` indirme,
  webcam ile tek satır test: `yolo track source=0 model=yolo26n.pt show=True`
- **B:** Herkesin ortamının (Python 3.11+) çalıştığını doğrulama.
- **D (kritik, yeni görev):** `API_SOZLESMESI.md`'yi frontend ekibine iletir;
  onların bu şemayla ilgili soru/itirazlarını toplayıp gün içinde netleştirir
  (örn. koordinat sistemi piksel değil metre olması onlar için sorun mu,
  `is_static` alanı yeterli mi).
- **C:** Hazır `backend/main.py` iskeletini kendi ortamında çalıştırıp
  `/api/health`, `/api/map/config` uç noktalarının doğru yanıt verdiğini
  doğrular; henüz gerçek kamera/model bağlanmadan mock `objects` listesiyle
  `/ws/tracking`'i test eder.

### Gün 2 — Tespit + takip pipeline'ı (tek kamera) + mock API testi
- **A:** `model.track(source=0, tracker="bytetrack.yaml", persist=True)` ile
  Python döngüsü kurma; her karede `id`, `class_name`, `bbox` çıktısını
  yakalama. Konsola yazdırarak doğrulama.
- **C:** Gün 1'deki mock API'yi frontend'e bağlayıp **sahte veriyle** ilk
  uçtan-uca görüntüyü alır — gerçek tespit verisi olmadan bile frontend'in
  "bir şey gösterdiğini" bu aşamada doğrulamak, sonraki entegrasyon riskini
  büyük ölçüde azaltır.
- **B:** Demo yapılacak fiziksel odanın/masanın netleştirilmesi, kamera
  konumunun (sabit, değişmeyecek) belirlenmesi.

### Gün 3-4 — Homografi kalibrasyonu ve harita izdüşümü
- **A:** Basit bir kroki görüntüsü (oda dış hatları, ölçekli) hazırlar ve
  `main.py`'deki `MAP_CONFIG.map_image_url`'in işaret ettiği `static/`
  klasörüne yerleştirir; gerçek `map_width_m`/`map_height_m` değerleriyle
  kodu günceller.
- **A:** Kamerada görünen 4 bilinen noktayı (örn. masa köşeleri, zemin
  deseni) ile krokideki gerçek metre karşılıklarını eşleştirip
  `CAMERA_POINTS`/`MAP_POINTS_METERS` değerlerini kendi ölçümleriyle
  günceller (`cv2.getPerspectiveTransform`/`findHomography` zaten iskelette
  hazır).
- **Doğrulama:** Biri odada yürür, konsolda yazdırılan `(x, y)` (metre
  cinsinden) değerlerinin mantıklı şekilde değiştiği gözlemlenir — API
  sözleşmesi zaten metre birimini varsaydığı için burada ek bir dönüşüm
  gerekmiyor.

### Gün 5 — Gerçek veri entegrasyonu (erken uçtan-uca test)
- **C:** A'nın kodunu bir arka plan thread'inde/process'inde çalıştırıp,
  Gün 1'de kararlaştırılan şemaya uygun gerçek veriyi (mock yerine) API
  üzerinden yayınlamaya başlar.
- **A + C + D birlikte (pairing, ~1 saat):** İlk gerçek uçtan-uca testi —
  kamera önünde biri hareket eder, frontend'de nokta hareket etmeli. Önceki
  plana göre bu test **2 gün erken** yapılıyor, çünkü sözleşme Gün 1'de
  netleşti.
- **B:** Bu testi izleyip gecikme/kopma gibi sorunları not eder.

### Gün 6 — Sağlamlaştırma ve görsel doğruluk
- **A:** Tespit eşiklerini (confidence threshold) demo odasının
  aydınlatmasına göre ince ayar yapar; yanlış pozitifleri azaltır.
- **C:** WebSocket bağlantı kopması durumunda otomatik yeniden bağlanma,
  hata durumunda anlamlı log/mesaj ekler.
- **D:** Frontend tarafında gerekiyorsa küçük ayarlar (harita ölçeği,
  ikon boyutu, sınıf-renk eşlemesi frontend'de zaten tanımlıysa onunla
  uyumluluk kontrolü).

### Gün 7-8 — 2. kamera füzyonu (artık stretch goal değil, hedeflenen özellik)
Frontend hazır olduğu için kazanılan zamanla bu artık "olursa iyi olur"
değil, **iki gün ayrılmış bir hedef:**
- İkinci bir kamera (telefon + "IP Webcam" Android veya "iVCam"/"EpocCam"
  iOS uygulamasıyla Wi-Fi üzerinden sanal webcam olarak) eklenir.
- Aynı odaya farklı açıdan bakan bu kamera için ayrı bir homografi
  kalibrasyonu (Gün 3-4'teki yöntemin tekrarı) yapılır.
- **Basit füzyon** (ana plandaki Macar algoritması yerine kısayol): İki
  kameradan gelen `(x, y)` koordinatları, belirli bir mesafe eşiğinin
  (örn. 0.5m) içindeyse "aynı kişi" kabul edilip tek bir noktada birleştirilir
  — tam MV3DT mantığının basitleştirilmiş, demo-yeterli hali.
- **Gün 8'in ikinci yarısı:** Bu füzyonun frontend'de doğru görünmesi için
  C+D pairing.

**Zaman yine daralırsa:** Bu iki gün, tek kameralı sürümü sağlamlaştırmaya
(farklı ışık koşulları, farklı kıyafet/nesnelerle test) ayrılır — artık
"acele mi yetişir" kaygısı yok, çünkü frontend işi baştan düşmüştü.

### Gün 9 — Genel prova ve son cila
- **Herkes:** Uçtan uca tam bir prova; §6'daki demo senaryosunun zamanlaması
  netleştirilir.
- **B:** Demo günü senaryosunu (bkz. §6) yazılı hale getirir, kim ne
  zaman ne yapacak netleştirir.

### Gün 10 — Prova ve yedek plan (buffer günü)
- **Herkes:** Sunum mekanında (veya en yakın koşullarda) **tam prova**.
- **B (kritik görev):** Canlı demo'nun **yedek kaydını** (screen recording)
  alır — canlı demo sırasında ışık/ağ/donanım sorunu çıkarsa bu video
  oynatılır. **Bu adımı asla atlamayın; canlı demo'larda donanım/ağ arızası
  istatistiksel olarak sık karşılaşılan bir durumdur.**
- Sunum akışı (bkz. §6) son kez zamanlanır (kaç dakika sürüyor, nerede
  duraklanacak).

---

## 5. Demo Senaryosu (Sunum Akışı)

1. **Açılış (30 sn):** Krokinin ekranda boş halde gösterilmesi — "bu bizim
   haritamız, şu an boş."
2. **Canlı tespit (1 dk):** Kamera görüntüsü ekranda, biri kadraja girer,
   tespit kutusu + ID anlık beliriyor.
3. **Harita eşleşmesi (1 dk):** Aynı anda krokide bir nokta beliriyor ve kişi
   hareket ettikçe nokta da hareket ediyor — **"asıl mesaj" burası.**
4. **Nesne çeşitliliği (30 sn):** Bir sandalye/laptop gösterilip onun da
   krokide farklı renkte işaretlendiği gösterilir.
5. **(Varsa) 2. kamera (1 dk):** İkinci kameranın aynı kişiyi farklı açıdan
   görüp krokide **tek bir nokta** olarak birleştiğinin gösterilmesi — bu,
   ana projenin en zor problemi olan "çakışan kameraların aynı yere
   baktığını anlaması" kavramının somut kanıtı.
6. **Kapanış (30 sn):** Bu demo'nun, sunulan 130 kameralı/çok katlı ana
   mimarinin (bkz. `proje_plani.md`) küçük ölçekli kanıtı olduğu vurgulanır.

**Toplam süre: ~4-5 dakika** — soru-cevap için ayrıca zaman bırakın.

---

## 6. B Planı (Frontend Ekibi Zamanında Yetişmezse)

Frontend ekibi kendi tarafında `API_SOZLESMESI.md`'ye göre geliştiriyor
olsa da, onların takvimi bizim kontrolümüzde değil. Gün 5'teki uçtan-uca
testte **asıl frontend hazır değilse**, demo'yu riske atmamak için:

- C, aynı `/ws/tracking` ve `/api/map/config` sözleşmesini tüketen **çok
  basit, geçici bir HTML/JS sayfası** (tek dosya, `<canvas>` ile nokta
  çizen ~50 satır kod) hazırlar — bu, asıl frontend'in yerini almaz, sadece
  demo günü bir "her ihtimale karşı" görselleştirici olarak durur.
- Bu geçici sayfa, sözleşme zaten sabit olduğu için **ekstra entegrasyon
  riski taşımaz** — API'yi değiştirmeden sadece ikinci bir tüketici eklemek
  demek.

**Bu kararı en geç Gün 6 sonunda verin** — asıl frontek ekibinden Gün 6'ya
kadar çalışan bir demo alınamadıysa, geçici sayfa Gün 7-8'de paralel
hazırlanmalı.

---

## 7. Demo'dan Ana Mimariye Köprü

Sunumdan sonra devam kararı alınırsa, demo bileşenlerinin ana plandaki
karşılıkları:

| Demo bileşeni | Ana plandaki karşılığı | Büyütme adımı |
|---|---|---|
| `ultralytics` tek model | DeepStream + YOLO26 custom parser + PeopleNet | DeepStream'e geçiş, çoklu model ayrımı |
| Homografi (4 nokta) | AMC / mimari plan tabanlı kalibrasyon | Çok kameralı, otomatikleştirilmiş kalibrasyona geçiş |
| Basit mesafe eşiği füzyonu | Macar algoritması + OSNet embedding (MV3DT) | Tam füzyon motoruna geçiş |
| Tek WebSocket | Kafka + Mosquitto + `bridge` servisi | Mesajlaşma altyapısına geçiş |
| Bellek-içi durum | PostgreSQL + `map-fusion-service` | Kalıcılık katmanı ekleme |
| Tek oda | Çok kat + çok kamera + Kubernetes | §2 ve §11'deki tam ölçekleme |

Demo kodu **çöpe atılmıyor** — `ultralytics`/homografi/WebSocket iskeleti,
ana mimarinin ilk gerçek entegrasyon testinde (örn. WILDTRACK ile test,
§Yol Haritası) referans olarak yeniden kullanılabilir.
