# vA Guard: hazır yüz tanımalı kroki takibinden güvenlik kararlarına

Tarih: 8 Ekim 2026. Ürün kapsamı: vA Guard Mining / kritik ve yüksek değerli alanlar.
Başlangıç: kullanıcının bildirdiği hazır yüz tanımalı kroki takip sistemi.
Bu belge Guard geliştirme planıdır. Yeni kurallar ve servisler henüz uygulanmış değildir.

## 1. Guard'da geliştireceğimiz yetenek

Hazır sistemin ürettiği kişi kimliği ve konum, Guard'ın karar girdisidir.
Sonraki hedef; kişinin bulunduğu bölgede, o zamanda ve o operasyon kapsamında
yetkili olup olmadığını, gerekli personelin mevcut olup olmadığını ve prosedür
adımlarının tamamlanıp tamamlanmadığını değerlendirmektir.

İşleme tesis içindeki NVIDIA sunucuda yapılır. Üretim çoklu kamera hattı
DeepStream kullanır. Birdeye/Spatial mevcut konum sağlayıcısı olarak tüketilir;
bu çalışmanın teslimatı Guard kural, prosedür, olay ve kanıt katmanıdır.

| Guard girdisi | Kullanım |
|---|---|
| Doğrulanmış kişi / kimlik kanıtı | Rol, ziyaretçi, vardiya ve operasyon yetkisini bulma |
| Kamera ve ortak takip kimliği | Kimliği hareket boyunca doğru kişiyle ilişkilendirme |
| Kroki konumu ve bölge | Gold Room, Vault, Search Area ve kontrollü koridor bağlamı |
| Zaman, vardiya, aktif operasyon | Süreli erişim ve operasyon bazlı izin |
| Veri sağlığı / konum kaynağı | Ölçüm, tahmin ve bilinmeyen durumu ayırma |
| Kapı, erişim, alarm ve operatör olayları | Prosedür ve fiziksel geçiş doğrulaması |

Yüz tanıma ve kroki kurulumunu yeniden geliştirmek başlangıç işi değildir.
Mevcut çıktılar Guard'a bağlanır; yalnızca gözlenen entegrasyon hataları giderilir.

## 2. EPFL ve bu reponun Guard içindeki görevi

`beratbaspinar/birdeyetech`, çoklu kamera takibi ve ortak zemin üzerinde iz
birleştirmesi sağlayan referanstır. Yüz kimlik servisi bu checkout'ta görülmüyor;
hazır ürünün kimlik çıktısı ayrıca tüketilecek.

EPFL Laboratory, aynı alanı gören dört kamerada kapanma, kamera geçişi, çift sayım
ve ID değişimini incelemek için kullanılır. Guard açısından bu sorunlar doğrudan
karar hatasına dönüşebilir: bir kişi iki iz olursa dual-person kuralı yanlış
sağlanabilir; başka kişiye aktarılan kimlik yanlış yetki kararı üretebilir.

EPFL kişi etiketleri gerçek çalışan kaydı/rol/izin değildir. Yüz doğruluğu,
liveness, yetkilendirme ve prosedür başarısı bu veriyle ölçülmez. EPFL takibi
üzerindeki Guard testleri sentetik rol ve bölge senaryoları olarak etiketlenir.
Gerçek ürün doğrulaması, hazır sistemin kontrollü pilotuyla yapılır.

Laboratory konumları `grid_cell` birimindedir; doğrulanmış metre ölçeği yoktur.
Tesis kuralları gerçek tesisin kalibrasyon ve harita birimlerini kullanır.
Araştırma veri kullanımından ticari ürün/gösterim dağıtım hakkı çıkarılmaz.

Önceki geliştirme paketinin Guard için destekleyici işleri:

- EPFL hücre merkezi dönüşümü ve etiket başlığı düzeltildi.
- GPU/video/model gerektirmeyen geometri kontrolü eklendi.
- Harita başlığı taşması ve kamera görüş poligonunun oda sınırını aşması düzeltildi.
- CPU test paketinde 623 test geçti.
- Resmi GT ile geometri kontrolünde 456/476 konum en az iki kamera çerçevesine
  düşüyor; bu takip, yüz doğruluğu veya Guard başarısı değildir.

[Geometri raporu](artifacts/epfl_geometry.json) kaynak dosya SHA-256 değerlerini taşır.
Tarihsel `epfl_demo.json` sonuçları eski evaluator'a aittir; tam takip benchmark'ı
yeniden çalıştırılmalıdır. Takip eşikleri bu çalışmada değiştirilmedi.

## 3. İlk ürün işi: Guard yetkilendirme ve bölge kural motoru

Önerilen ilk servis `GuardDecisionEngine` olur. Kişi ve konum gözlemlerini,
bölge politikası ile aktif operasyon bağlamına göre değerlendirir.

| Kural | Guard davranışı | Doğrulama senaryosu |
|---|---|---|
| Kişi + rol + bölge + zaman + operasyon | İzinli giriş, yetkisiz giriş veya doğrulama gereksinimi | Gold Room'a yetkili kişi girer; Vault yetkisi yoksa Vault girişinde olay oluşur |
| Süreli ziyaretçi / bakım izni | İzin penceresi ve izinli bölgeleri kontrol eder | Süresi biten ziyaretçi içeride kalır veya farklı bölgeye geçer |
| Yasak bölgeye giriş | Tek olay kaydı ve operatör bildirimi üretir | Sınır çevresindeki konum titreşimi aynı olay için sürekli alarm üretmez |
| Bekleme ve rota | Yapılandırılmış bekleme süresi / izinli rota dışına çıkmayı kontrol eder | Ziyaretçi onaylı koridordan sapar |
| Kimlik / konum belirsizliği | Kararı UNKNOWN / doğrulama gerekli olarak yayınlar | Kimliği doğrulanmayan kişinin durumu otomatik olarak yetkisiz ilan edilmez |

Kimlik kanıtı, kaynak oturumu ve iz bağı korunur. Global takip numarası çalışan
kimliği yerine kullanılmaz. Çelişen kimlik eşlemeleri sessizce birleştirilmez.
Kişinin yüzü kaybolduğunda korunmuş takip bağı ile son doğrulama kanıtı kullanılır;
iz bağı belirsizleştiğinde yeniden doğrulama gereksinimi oluşur.

Önerilen Guard olay sözleşmesi:

```json
{
  "schema_version": 1,
  "event_id": "example-event-001",
  "event_type": "UNAUTHORIZED_ZONE_ENTER",
  "event_time": "2026-10-08T20:18:00Z",
  "person_id": "example-person",
  "tracking_session_id": "example-run",
  "global_track_id": 17,
  "zone_id": "VAULT_ZONE",
  "operation_id": "example-operation",
  "policy_id": "example-vault-access",
  "policy_version": 1,
  "decision": "deny",
  "reason_codes": ["role_not_allowed_for_zone"],
  "evidence_refs": ["example-identity-evidence", "example-position-evidence"],
  "camera_refs": [{"camera_id": "CAM_01", "session_id": "example-source"}],
  "data_health": "valid",
  "ack_status": "unacknowledged"
}
```

Bu örnek bir tasarımdır; mevcut API veya ölçülmüş sonuç değildir. İlk sürüm karar,
olay, kanıt ve operatör akışını kapsar. Fiziksel kapı kontrolü, bağlantı ve
yetkilendirme sözleşmesi sağlandıktan sonra ayrıca entegre edilir.

## 4. İkinci iş: dual-person ve eskort sürekliliği

Dual-person kuralı, gerekli bölgede ve aktif operasyonda en az iki farklı,
doğrulanmış ve uygun yetkili kişinin bulunmasını kontrol eder. İki kamera kutusu
veya iki takip numarası tek başına iki farklı yetkili kişi kanıtı değildir.

İlk demoda yapılandırılmış operasyon için iki kişi mevcutken durum normaldir;
biri ayrıldığında, politika tolerans süresinden sonra `DUAL_CONTROL_BROKEN`
olayı oluşur. Süreler pilotta belirlenir; bu belgede doğrulanmış eşik ilan edilmez.
Kamera kaybı nedeniyle ikinci kişinin görülmemesi, onun gerçekten ayrılmasıyla
aynı değerlendirilmez; gözlem yetersizliğinde durum UNKNOWN olur.

Eskort kuralı, ziyaretçi ve atanmış eskortun aynı izinli yolculuk bağlamında
kalmasını izler. Takip belirsizliği ve görüş boşlukları ayrı nedenler olarak
kaydedilir; yalnızca görünüş benzerliğinden çalışan kimliği atanmaz.

## 5. Üçüncü iş: prosedür ve olay kanıtı

Guard yolculuğu Secure Entrance → Search Area → Controlled Corridor → Gold Room
→ Vault olarak modellenir. Geçiş sırası aktif görev ve politika ile ilişkilendirilir.

Search Area'da bulunmak, fiziksel aramanın yapıldığını tek başına kanıtlamaz.
Arama/checklist adımı operatör veya uygun harici sistem olayıyla tamamlanır.
Gold pour, sevkiyat ve lock-up gibi operasyonlar ayrı durum makineleri olur;
hangi adımın hangi sensör/operatör kanıtıyla doğrulandığı açıkça tanımlanır.

Her olay; kişi, bölge, politika/operasyon sürümü, zaman, kaynak kameralar, kimlik
kanıtı, konum kalitesi ve yerel video referansıyla kaydedilir. Operatör ACK,
olay açıklaması ve yapılandırılmış escalation akışı aynı kayıt üzerinde tutulur.
Tartı/varlık/kapı entegrasyonları bu kayıt modeline sonradan bağlanır.

## 6. DeepStream ve yerel işletim

DeepStream kamera işleme çıktısını Guard'ın mevcut kişi-konum girişine bağlayan
adaptör hazırlanır. Kamera/oturum/iz anahtarı, kaynak zamanı, kare yaşı,
kalibrasyon sürümü ve ölçüm/tahmin ayrımı korunur. Tüm işlemler yerel sunucuda
çalışır; API/veritabanı işleri görüntü callback'ini bloke etmez.

Kamera yeniden bağlantısı yeni oturum açar. Eski kare güncel konum gibi sunulmaz.
Kamera veya kalibrasyon sorunu ilgili Guard bölgesini UNKNOWN yapar; boş alan
veya prosedürün tamamlandığı sonucu üretilmez.

## 7. İlk Guard demosunun kabulü

| Senaryo | Beklenen çıktı |
|---|---|
| Yetkili giriş | Yetki bağlamı ve yolculuk kaydı; ihlal olayı yok |
| Yetkisiz Vault girişi | Kişi-bölge-politika nedeni ve kanıt içeren tek ihlal olayı |
| Kimliği belirsiz kişi | Kimlik doğrulama gereksinimi; uydurulmuş kişi/yetki yok |
| Dual-person bozulması | İkinci gerçek yetkili kişi ayrıldıktan sonra politika bazlı olay |
| Kamera geçişi / kısa kapanma | Yanlış kişiye yetki aktarımı ve çift sayım ölçülür |
| Kamera kaybı / eski konum | Bölge veya karar UNKNOWN; veri sağlığı nedeni görünür |
| Prosedür adımı eksik | Eksik adım ve eksik kanıt gösterilir |
| Tekrarlı gözlem / olay replay | Aynı olay tekrar tekrar bildirilmez; politika sürümüyle yeniden oynatılır |

İlk teslimat paketi: mevcut kişi/konum çıktısı için Guard giriş adaptörü,
yetkilendirme/bölge kural motoru, olay/kanıt kaydı ve operatör olay ekranı.
Ardından dual-person/eskort ve prosedür durum makineleri gelir. EPFL bu teslimatın
çoklu kamera takip doğrulamasını destekler; ürünün merkezi Guard karar katmanıdır.

## Kaynaklar ve proje girdileri

- Kullanıcının mevcut yüz tanımalı kroki takip sistemi ve 8 Ekim Guard kapsam düzeltmesi.
- Kullanıcının vA Guard Mining sunumu: kimlik, bölge, prosedür, erişim, varlık ve yanıt bağlamı.
- Kullanıcının 23 Eylül yerel sunucu ve DeepStream mimari kararı.
- [EPFL veri seti / kalibrasyon / GT referansı](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/)
- [Repo takip sözleşmeleri](../src/mcreid/fusion/types.py)
- [EPFL geometri raporu](artifacts/epfl_geometry.json)
- [WILDTRACK takip hata analizi](wildtrack_results.md)
