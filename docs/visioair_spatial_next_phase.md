# Birdeye / Spatial: EPFL'den Guard entegrasyonuna

Tarih: 8 Ekim 2026. İncelenen temel commit: `2439cc501e8f77b80361df7e442240408af60652`.
Bu belge geliştirme önerisidir; tamamlanan kod ile planlanan entegrasyonu ayrı belirtir.

## 1. Başlangıç noktası

Emir'in güncel bildirimine göre yüz tanımalı kroki takibi hazır. Bu yeteneği mevcut
kimlik servisi olarak kullanacağız. Bu GitHub reposunda görülen kod ise çoklu kamera
beden Re-ID'si ve ortak zemin takibi; yüz kayıt/verifikasyon servisi bu checkout'ta yok.
İki sistemi bağlamak için servis çıktısının sözleşmesi gerekir, yüz tanımayı yeniden
geliştirmek gerekmez.

| Repo bileşeni | Mevcut görev | Sonraki entegrasyondaki rol |
|---|---|---|
| `calib/epfl.py`, `scripts/fetch_epfl.py` | EPFL kalibrasyonu / etiketleri / indirme | Araştırma ve regresyon yüzeyi |
| `track/gpu_view.py`, `track/per_view.py` | Kişi tespiti, beden embedding'i, kamera içi takip | Referans algılama sağlayıcısı |
| `fusion/global_id.py`, `fusion/motion.py` | Geometri, hareket ve görünüş ile ortak takip | Birdeye takip çekirdeği |
| `fusion/types.py` | View/Ground observation ve global snapshot | Spatial çıktı adaptörünün girdisi |
| `cli/epfl_live.py`, `cli/public_demo.py` | Ardışık canlı izleme / seyrek etiketli değerlendirme | Demo ve benchmark ayrı yollar |
| `calib/homography.py`, `calib/floor.py` | Fiducial/nokta tabanlı zemin kalibrasyonu | Gerçek tesis commissioning sağlayıcısı |
| `viz/bev.py` | Ortak zemin haritası | Operatör haritasına referans |

23 Eylül mimari kararları korunur: işlemler tesis içindeki NVIDIA sunucuda çalışır;
üretim görüntü hattı DeepStream olur; Birdeye genel amaçlı Spatial çekirdeğidir;
Guard kimlik, yetki ve prosedür bağlamını üst katmanda ekler. Normal çalışmada
internet bağlantısı veya çalışma sırasında model indirme gerekmemelidir.

## 2. EPFL'nin sağlayacağı kanıt

İlk deney Laboratory 6-person kaydıdır: aynı alanı gören dört senkron kamera,
25 FPS video, kamera başına kalibrasyon ve seyrek kişi-konum etiketleri. Şunları
incelemek için uygundur: örtüşen kameralar arasında tek ortak iz, çift sayım,
ID değişimi, kapanma sonrası yeniden yakalama ve konum tutarlılığı.

Resmi GT kimlikleri kişi sütunlarıdır; yüz kimliği veya çalışan adı değildir.
Bu deney yüz tanıma doğruluğunu, liveness'ı, rol yetkisini veya tesis erişim
kararlarını doğrulamaz. Bunlar hazır yüz servisi ve kontrollü pilot verisiyle
ayrı test edilir.

Laboratory homografisi görüntüden top-view'a dönüşümdür. Top-view 358×360,
GT ızgarası 56×56'dır. Resmi `grid_to_tv` konumu hücre merkezinde tanımlar:

```text
column = position_id % grid_width
row = position_id // grid_width
tv_x = (column + 0.5) * tv_width / grid_width + tv_origin_x
tv_y = (row + 0.5) * tv_height / grid_height + tv_origin_y
```

Bu değişiklikte köşe yerine merkez kullanıldı; instrument ve benchmark aynı
dönüşüm fonksiyonunu tüketir. GT başlığındaki `step_size` kare aralığıdır;
video FPS değildir. Negatif etiketler (`-1` bilinmiyor, `-2` grid dışında)
konuma çevrilmez. Bozuk başlık, eksik satır ve geçersiz grid indeksleri reddedilir.
Yeni JSON sonuçları `ground_truth_convention: cell_center_v1` taşır.

Resmi iki metin dosyasıyla yeni geometri kontrolü yapıldı: 476 etiketli konumun
456'sı (%95,8) en az iki kamera çerçevesine izdüşüyor. [Yeni geometri kaydı](artifacts/epfl_geometry.json)
kaynak SHA-256 değerlerini taşır; dedektör uyumu, ID sürekliliği ve yüz doğruluğu
bu kontrolde çalıştırılmadı. Geometri aşaması video/model/GPU olmadan tekrar edilir:

```bash
uv run python scripts/check_epfl_instrument.py --geometry-only
```

Laboratory için doğrulanmış metre ölçeği bu repoda yoktur. Sonuç birimi `grid_cell`
kalır. Dosya/API adlarındaki `_m` ekleri EPFL'yi fiziksel metre yapmaz. Gerçek tesis
konumu ise ölçüsü doğrulanmış ortak harita ve gerçek kamera kalibrasyonuyla `m`
biriminde yayınlanır. Zemin homografisi tek başına tam 3D hacim rekonstrüksiyonu
veya insan yüksekliği üretmez; bu ayrı bir geliştirme ve doğrulama konusudur.

EPFL sayfası video, kalibrasyon ve etiketler için araştırma kullanımını belirtir.
Bu veri Ar-Ge değerlendirmesinde kullanılacak; ürün paketine veya ticari müşteri
gösterimine ekleme hakkı bu ifadeden çıkarılmayacak. Kaynak veriler/kişi crop'ları
bu değişiklikle repoya eklenmez.

## 3. Önce ölçüm zemini, ardından takip iyileştirmesi

Commit'teki tarihsel `epfl_demo.json`, 60 seyrek karede calibrated kol için 27
ID değişimi ve yaklaşık 1.699 hücre ortalama konum hatası bildiriyor. Bunlar eski
hücre-köşesi evaluator'ına aittir. Yeni düzeltmenin iyileştirme sağladığına dair
benchmark yapılmadı; eski JSON değerleri değiştirilmiyor.

İlk karşılaştırma aynı videolar, kareler, dedektör ve Re-ID modeliyle tekrar edilir:

```bash
uv sync --extra dev
uv run pytest tests/test_calib_epfl.py tests/test_epfl_live.py
uv run python scripts/fetch_epfl.py
uv run python scripts/check_epfl_instrument.py
uv run mcreid-public-demo epfl --stages all
```

Araştırma sayılarından ürün hedefi türetilmez. Mevcut eşleme/merge eşikleri bu PR'da
değişmez. Yeni baselinedan sonra her denemede tek ana değişken değiştirilir.

| Ölçüm | Raporlama |
|---|---|
| Konum hatası | Ortalama, p50, p90/p95; EPFL'de hücre, metrik pilotta metre |
| Kimlik sürekliliği | ID switch, iz parçalanması, uzun kayıp sonrası yanlış kişiye bağlanma |
| Sayım | Kaçırma, yanlış pozitif, çift sayım ve kişi başına çoklu iz ayrı ayrı |
| Gecikme | Kaynak zamanından Spatial yayınına p50/p95; kamera başına işlenen FPS |
| Kaynak yükü | GPU, VRAM, decode yükü, kuyruk yaşı ve düşürülen kare sayısı |
| Veri kalitesi | Geçersiz konum, eksik kamera, eski kare ve kalibrasyon durumu |

Seyrek GT benchmark'ı, 25 FPS ardışık takipten farklıdır. Gerçek kaynak zamanları
korunmalı; atlanan/çözülemeyen karelerde hareket modeli gerçek geçen süreyi
kullanmalıdır. Uzun kayıp/kimlik geri çağırma sonuçları yalnız birkaç seed'e veya
tek kolay kişiye dayandırılmamalıdır. README'deki adversarial long-gap hatası
ayrı regresyon senaryosu olarak kalır.

## 4. Hazır yüz servisini ortak takibe bağlamak

Üç kimlik ayrı tutulur:

| Kimlik | Anlam | Örnek |
|---|---|---|
| Kamera izi | Kaynağın belirli oturumundaki iz | `camera_id + session_id + local_track_id` |
| Global iz | Birdeye'nin ortak haritadaki takip nesnesi | `tracking_session_id + global_track_id` |
| Kişi kimliği | Yüz servisiyle doğrulanan kayıt | `person_id`, doğrulama yoksa `null` |

Global takip numarası çalışan kaydı değildir. Yüz görünmediğinde beden görünüşü,
konum ve hareket sürekliliği izi taşıyabilir; yeni yüz doğrulaması yapılmış gibi
etiketlenmez. Eşleme belirsizleşince kişi bağı düşürülür; yeni kaliteli yüz sonucu
yeniden bağlar. Çelişen kişi kimlikleri sessizce tek kişiye birleştirilmez.

Önerilen kimlik adaptörü `person_id`, `verification_event_id`, kamera/oturum/iz
anahtarı, kanıt zamanı, yüz kalitesi ve doğrulama skorunu alır. Skorun kalibre
olasılık olduğu varsayılmaz. Kimlik kanıtı, aynı kaynaktaki kişi kutusuna zamansal
ve geometrik olarak bağlanır; bir karedeki başka kişinin yüzü iz üzerine aktarılmaz.
Liveness durumu sadece mevcut yüz servisinde ölçülüyorsa yayınlanır.

Önerilen SpatialEntity sözleşmesi:

```json
{
  "schema_version": 1,
  "map_id": "MAP_01",
  "coordinate_frame_id": "MAP_01_FLOOR",
  "tracking_session_id": "example-run",
  "global_track_id": 17,
  "person_id": null,
  "identity_status": "unknown",
  "identity_evidence_time": null,
  "event_time": "2026-10-08T19:55:00Z",
  "timestamp_source": "source_pts",
  "source_track_refs": [
    {"camera_id": "CAM_01", "session_id": "example-source", "local_track_id": 24}
  ],
  "position": {"xy": [2.1, 4.3], "unit": "m", "source": "measured"},
  "position_covariance": {"matrix": [[0.04, 0.0], [0.0, 0.09]], "unit": "m2"},
  "track_state": "confirmed",
  "calibration_refs": [{"camera_id": "CAM_01", "calibration_id": "example-cal-v1"}],
  "quality_reason": "valid",
  "frame_age_ms": 85
}
```

Bu örnek yeni API tasarımıdır, mevcut endpoint veya ölçülmüş sonuç değildir.
EPFL adaptörü `grid_cell` / `grid_cell2` birimlerini kullanır. Geçersiz veya eski
konumda `position: null` ve neden yayınlanır. COASTING tahminleri `source: predicted`
olarak etiketlenir; gözlenen dolulukla aynı şey sayılmaz. Süre sınırları pilotta
ölçülerek yapılandırılır, bu belgede doğrulanmış değer ilan edilmez.

## 5. DeepStream ve Guard bağlantısı

Mevcut Python perception hattı referans olarak kalır. Üretim hattında aynı
Spatial sözleşmesini üreten bir DeepStream metadata sağlayıcısı hazırlanır:
RTSP → donanımsal decode → nvstreammux → nvinfer → nvtracker → metadata adaptörü
→ Birdeye fusion → SpatialEntity → zone/event → Guard.

DeepStream metadata bbox/temas koordinatları, resize/padding/crop ve lens
profilinden sonra kalibrasyonun piksel uzayına dönüştürülür. Re-ID özelliğinin
üretim modeli, normalizasyonu ve boyutu referans backend ile açıkça eşleştirilir;
tracker object-ID'lerinin kameralar arasında global olduğu varsayılmaz.

Adaptör metadata'yı sınırlı kuyruğa kopyalar ve yerel IPC ile aktarır. Kamera
yeniden bağlanınca yeni `session_id` açılır. API/veritabanı işlemleri görüntü
callback'ini bloke etmez. Model ağırlıkları kurulum paketinde yerelde bulunur.

İlk generic zone motoru ENTER/EXIT/DWELL, doluluk ve rota geçmişini yayınlar.
Bölge sınırındaki konum belirsizliği debounce/hysteresis ile ele alınır. Kamera
kesintisi veya STALE kalibrasyon ilgili bölgeyi UNKNOWN yapar; boş alan üretmez.
Guard; kişi + rol + bölge + zaman + operasyon bağlamını ekler. İlk Guard senaryosu
yasak bölgeye giriş, kanıt kaydı ve operatör bildirimi olur. Dual-person, escort,
kapı/tartı ve prosedür füzyonu doğrulanmış bu çekirdeğin sonraki tüketicileridir.

## 6. Uygulama sırası ve kabul kanıtı

| Sıra | İş paketi | Tamamlanma kanıtı |
|---|---|---|
| 1 | EPFL referans dönüşümü ve GT parser | Bu değişiklikte kod/test; tam benchmark yeniden çalıştırılacak |
| 2 | SpatialEntity adaptörü ve mevcut yüz servisi bağlama | Aynı kişiye çoklu kamera desteği; yüz yokken yeni doğrulama üretmeme; çelişki testi |
| 3 | Generic zone/trajectory/event replay | ENTER/EXIT/DWELL deterministik replay; UNKNOWN ve predicted ayrımı |
| 4 | DeepStream metadata sağlayıcısı | Referans ile aynı koordinat/kimlik sözleşmesi; kesinti izolasyonu ve gecikme ölçümü |
| 5 | Gerçek ofis / tesis pilotu | Ölçülü metrik harita, tutulmuş doğrulama noktaları, kimlik/zone senaryoları |
| 6 | Guard adaptörü | Kişi-yetki-bölge-zaman kararı, olay kanıtı ve operatör akışı |

Ekli ofis fotoğrafları ve KatPlanı ölçülü kalibrasyon değildir: haritada ölçüler,
görüntü-kamera ilişkisi ve kaynak dönüşümleri ayrıca doğrulanır. Masaların arkasında
ayaklar kapanınca bbox altını zemindeki gerçek temas kabul etmek hata üretir.
Geçerli zemin alanı ve görüş desteği raporlanmalı; görünmeyen alan da kapsanmış gibi
gösterilmemelidir. Ürün commissioning'i otomatik destekli/fiducial akışa dayanır;
tesis başına yüzlerce manuel düzeltme veya ana kullanıcı akışında uzun nokta işaretleme
işlemi hedeflenmez. Başlangıçta gerçek XYZ triangulation için EPFL'nin nominal K'sı
kullanılmaz; gerekiyorsa gerçek intrinsics/extrinsics ve ayrı doğrulama sağlanır.

İlk hedef yeni bir model eğitmek değil, hazır kimlik/kroki ürününü bu repo ile
ölçülebilir şekilde birleştirmektir. Yeni model kararı, düzeltilmiş baseline ve
pilot hata analizi hangi bileşenin sınırladığını gösterdikten sonra alınır.

## Kaynaklar

- [EPFL CVLab veri seti, calibration ve GT referansı](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/)
- [Laboratory 6-person kalibrasyonu](https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/calibration-6p.txt)
- [Laboratory 6-person GT](https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/gt_lab_6p.txt)
- [Mevcut EPFL ölçüm kaydı](artifacts/epfl_demo.json) ve [instrument kaydı](artifacts/epfl_instrument.json)
- [Mevcut WILDTRACK hata analizi](wildtrack_results.md)
- [Mevcut takip sözleşmeleri](../src/mcreid/fusion/types.py)
- Kullanıcının 23 Eylül yerel sunucu/DeepStream mimari belgesi ve 8 Ekim hazır yüz tanımalı kroki takibi bildirimi.
