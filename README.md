# Türkçe Hikaye Dil Modeli

Bu proje, Türkçe hikayelerden sıfırdan eğitilen küçük bir dil modeli laboratuvarıdır. Amaç iki yönlüdür:

1. Dil modeli eğitiminin temel fikirlerini temiz ve okunabilir kodla öğrenmek.
2. Türkçe hikaye üretimi üzerine bir araştırma bildirisi için deney zemini kurmak.

Projede iki model vardır:

- **N-gram dil modeli**: Bağımlılıksız, şeffaf ve hızlı bir temel model. Karakterleri önceki karakterlere bakarak tahmin eder.
- **Mini Transformer**: PyTorch ile yazılmış küçük bir sinir ağı modeli. Aynı veri üzerinde n-gram yaklaşımıyla karşılaştırma yapmak için eklenmiştir.

Yerel web arayüzü sayesinde modelleri tarayıcıdan deneyebilir, prompt yazıp hikaye üretimi yapabilirsiniz.

## Kurulum

Python 3.10+ yeterlidir.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

Transformer deneyleri için PyTorch gerekir:

```bash
python -m pip install -e .[transformer]
```

Paket kurmadan da komutları `python scripts/...` şeklinde çalıştırabilirsiniz. Betikler kendi içlerinde `src/` klasörünü Python yoluna ekler.

## Hızlı Başlangıç

N-gram modelini mini Türkçe hikaye derlemiyle eğitin:

```bash
python scripts/train_ngram.py --data data/stories/tr_mini_stories.txt --out runs/ngram_tr.json --order 7 --alpha 0.001
```

Ömer Seyfettin hikayeleriyle daha büyük n-gram modelini eğitin:

```bash
python scripts/train_ngram.py --data data/stories/omer_seyfettin_hikayeleri.txt --out runs/ngram_omer_seyfettin.json --order 7 --alpha 0.001
```

N-gram ile metin üretin:

```bash
python scripts/generate.py --model runs/ngram_tr.json --prompt Bir sabah --length 600 --temperature 0.7 --top-k 8 --out runs/ornek_hikaye.txt
```

Ömer Seyfettin modeliyle metin üretin:

```bash
python scripts/generate.py --model runs/ngram_omer_seyfettin.json --prompt Bir sabah --length 600 --temperature 0.75 --top-k 12 --out runs/omer_seyfettin_ornek_hikaye.txt
```

N-gram modelini değerlendirin:

```bash
python scripts/evaluate.py --model runs/ngram_tr.json --data data/stories/tr_mini_stories.txt
```

Transformer eğitin:

```bash
python scripts/train_transformer.py --data data/stories/tr_mini_stories.txt --out runs/transformer_tr.pt --steps 500
```

Transformer ile metin üretin:

```bash
python scripts/generate_transformer.py --model runs/transformer_tr.pt --prompt Bir sabah --length 600 --temperature 0.8 --top-k 8 --out runs/transformer_ornek_hikaye.txt
```

Transformer'ı değerlendirin:

```bash
python scripts/evaluate_transformer.py --model runs/transformer_tr.pt --data data/stories/tr_mini_stories.txt
```

Yerel web arayüzünü açın:

```bash
python scripts/serve_web.py --port 7860
```

Tarayıcıda `http://127.0.0.1:7860` adresine gidin.

Sunucu açıkken hızlı API testi:

```bash
python scripts/smoke_web.py
```

## Veri Akışı

Projedeki temel akış şöyledir:

```text
data/stories/tr_mini_stories.txt
        |
        v
scripts/train_ngram.py veya scripts/train_transformer.py
        |
        v
runs/ngram_tr.json veya runs/transformer_tr.pt
        |
        v
scripts/generate*.py, scripts/evaluate*.py veya web arayüzü
```

Veri dosyaları paragraflara ayrılmış kısa Türkçe hikayelerden oluşur. N-gram modelinde bu paragraflar ayrı hikayeler olarak ele alınır. Transformer modelinde dosyanın tamamındaki karakter dizisi eğitim verisi olarak kullanılır.

## Proje Yapısı

- `data/stories/tr_mini_stories.txt`: Başlangıç için kısa, özgün Türkçe hikayeler.
- `data/stories/omer_seyfettin_hikayeleri.txt`: Ömer Seyfettin hikayelerinden oluşan daha büyük eğitim dosyası.
- `src/turkish_story_lm/tokenizer.py`: Karakter düzeyi tokenizer.
- `src/turkish_story_lm/ngram.py`: Sıfırdan yazılmış n-gram dil modeli.
- `src/turkish_story_lm/transformer.py`: Küçük karakter düzeyi Transformer modeli.
- `scripts/train_ngram.py`: N-gram eğitim betiği.
- `scripts/generate.py`: N-gram hikaye üretim betiği.
- `scripts/evaluate.py`: N-gram için ortalama negatif log olabilirlik ve perplexity.
- `scripts/train_transformer.py`: PyTorch Transformer eğitim betiği.
- `scripts/generate_transformer.py`: Transformer ile hikaye üretimi.
- `scripts/evaluate_transformer.py`: Transformer için kayıp ve perplexity.
- `scripts/serve_web.py`: Yerel web arayüzü ve JSON API.
- `scripts/smoke_web.py`: Yerel web API için hızlı kontrol.
- `web/`: Tarayıcıda çalışan hikaye üretim arayüzü.
- `research/bildiri_zemini.md`: Araştırma bildirisi için problem, deneyler ve taslak.
- `research/deney_notlari.md`: İlk deney sonuçları ve yorumlar.
- `runs/`: Eğitilmiş modeller ve üretilmiş örnekler.

## Kodlar Nasıl Çalışıyor?

### `tokenizer.py`

Bu dosya karakter düzeyinde basit bir tokenizer tanımlar.

Temel fikir:

- `CharTokenizer.train(text)` metindeki benzersiz karakterleri bulur.
- `encode(text)` her karakteri sayısal kimliğe çevirir.
- `decode(token_ids)` sayısal kimlikleri tekrar metne çevirir.
- Türkçe karakterler ayrı işleme tabi tutulmaz; Python stringleri Unicode desteklediği için doğrudan korunur.

Bu proje sözcük veya alt-sözcük tokenizer kullanmaz. Bunun nedeni, dil modeli mantığını en sade haliyle göstermektir.

### `ngram.py`

Bu dosya karakter düzeyi n-gram dil modelini içerir.

N-gram modeli şu soruyu cevaplar:

```text
Önceki n-1 karakteri gördüğümde sıradaki karakter ne olabilir?
```

Örnek:

```text
Bağlam: "Bir saba"
Sıradaki karakter adayları: "h", "n", "r", ...
```

Modelin ana bölümleri:

- `fit(texts)`: Eğitim metinlerinden bağlam-karakter sayımları çıkarır.
- `next_token_distribution(context)`: Verilen bağlam için sonraki karakter olasılıklarını hesaplar.
- `generate(...)`: Olasılıklara göre karakter seçerek metni uzatır.
- `negative_log_likelihood(text)`: Modelin bir metni ne kadar iyi tahmin ettiğini ölçer.
- `perplexity(text)`: Dil modellemede yaygın kullanılan okunabilirlik/olasılık metriğini verir.

Model her karakter için yalnızca en uzun bağlamı değil, daha kısa bağlamları da kaydeder. Bu önemlidir; çünkü kullanıcı eğitim verisinde aynen geçmeyen bir prompt yazdığında model tamamen rastgele seçime düşmez. Uzun bağlam bulunamazsa daha kısa bağlamlarla tahmin yapar.

`alpha` parametresi additive smoothing içindir. Yani modelin az görülen veya hiç görülmeyen karakterlere sıfır olasılık vermesini engeller.

`top_k` parametresi üretim sırasında sadece en olası `k` karakter arasından seçim yapar. Bu, küçük veriyle çalışırken anlamsız noktalama veya rastgele karakter çıkma riskini azaltır.

### `transformer.py`

Bu dosya küçük bir causal Transformer dil modeli tanımlar.

Modelin temel parçaları:

- `TransformerConfig`: Model boyutlarını tutan ayar sınıfı.
- `TinyTransformerLM`: PyTorch `nn.Module` olarak yazılmış dil modeli.
- `token_embedding`: Karakter kimliklerini vektöre çevirir.
- `position_embedding`: Karakterlerin sıradaki konum bilgisini modele ekler.
- `TransformerEncoderLayer`: Self-attention ve feed-forward katmanları.
- `head`: Son vektörleri karakter olasılıklarına çeviren lineer katman.

Bu model causal maske kullanır. Yani model, sıradaki karakteri tahmin ederken gelecekteki karakterleri göremez. Bu davranış dil modeli eğitimi için gereklidir.

`generate(...)` fonksiyonu n-gram modelindeki gibi prompttan başlar, sonra karakterleri tek tek üretir. Sıcaklık ve top-k ayarları burada da kullanılır.

Mini veri kümesinde Transformer'ın n-gramdan daha zayıf görünmesi normaldir. Transformer daha esnek bir modeldir, fakat iyi sonuç için daha fazla veri ve daha uzun eğitim ister.

## Betikler

### `scripts/train_ngram.py`

Bu betik n-gram modelini eğitir.

İş akışı:

1. `--data` ile verilen hikaye dosyasını okur.
2. Boş satırlara göre hikayeleri ayırır.
3. `NGramLanguageModel(order, alpha)` oluşturur.
4. `fit(stories)` ile karakter bağlamlarını sayar.
5. Modeli `--out` konumuna JSON olarak kaydeder.

Çıktı dosyası insan tarafından okunabilir JSON'dur. Bu, n-gram modelin nasıl sayım yaptığını incelemek için faydalıdır.

### `scripts/generate.py`

Bu betik eğitilmiş n-gram modelinden hikaye üretir.

Önemli parametreler:

- `--prompt`: Hikayenin başlangıç metni.
- `--length`: Üretilecek en fazla yeni karakter sayısı.
- `--temperature`: Düşük değer daha güvenli, yüksek değer daha cesur üretim verir.
- `--top-k`: Her adımda en olası kaç karakter arasından seçim yapılacağı.
- `--seed`: Aynı ayarlarla tekrar edilebilir üretim sağlar.
- `--out`: Sonucu dosyaya kaydeder.

### `scripts/evaluate.py`

Bu betik n-gram modelini veri dosyası üzerinde değerlendirir.

Raporladığı değerler:

- `mean_nll`: Ortalama negatif log olabilirlik. Düşük olması daha iyidir.
- `mean_perplexity`: Modelin tahmin belirsizliğini özetler. Düşük olması daha iyidir.

Mini veri üzerinde çok düşük perplexity ezberleme anlamına gelebilir. Bu nedenle metrikleri insan değerlendirmesiyle birlikte yorumlamak gerekir.

### `scripts/train_transformer.py`

Bu betik Transformer modelini eğitir.

İş akışı:

1. Veri dosyasını okur.
2. `CharTokenizer` ile karakterleri sayısal kimliklere çevirir.
3. Rastgele mini-batch'ler oluşturur.
4. Modelin sıradaki karakteri tahmin etmesini ister.
5. Cross entropy loss ile hatayı hesaplar.
6. AdamW optimizer ile ağırlıkları günceller.
7. Modeli `runs/transformer_tr.pt` dosyasına kaydeder.

Kaydedilen `.pt` dosyası şunları içerir:

- Model ayarları.
- Tokenizer sözlüğü.
- Model ağırlıkları.
- Eğitim meta verisi.

### `scripts/generate_transformer.py`

Bu betik eğitilmiş Transformer modelinden metin üretir. `generate.py` ile benzer parametrelere sahiptir, fakat model dosyası `.pt` formatındadır.

### `scripts/evaluate_transformer.py`

Bu betik Transformer modelini sabit uzunluklu parçalar üzerinden değerlendirir. Her parça için sıradaki karakter tahmin kaybı hesaplanır ve ortalaması raporlanır.

### `scripts/serve_web.py`

Bu betik yerel web uygulamasını başlatır.

İki iş yapar:

- `web/` klasöründeki HTML, CSS ve JavaScript dosyalarını sunar.
- `/api/generate` ve `/api/models` endpoint'lerini sağlar.

Endpointler:

```text
GET  /api/models
POST /api/generate
```

`POST /api/generate` şu alanları alır:

```json
{
  "model": "ngram",
  "prompt": "Bir sabah",
  "length": 600,
  "temperature": 0.7,
  "top_k": 8,
  "seed": 7
}
```

### `scripts/smoke_web.py`

Sunucu açıkken API'nin çalışıp çalışmadığını hızlıca kontrol eder. Önce modellerin mevcut olup olmadığını sorar, sonra kısa bir n-gram üretim isteği yollar.

## Web Arayüzü

`web/` klasörü tek sayfalık bir arayüz içerir.

- `web/index.html`: Sayfa iskeleti ve form alanları.
- `web/styles.css`: Görsel tasarım.
- `web/app.js`: Formdan verileri okur, API'ye istek atar ve sonucu ekrana yazar.

Arayüzde şu ayarlar değiştirilebilir:

- Prompt.
- Model seçimi.
- Üretim uzunluğu.
- Sıcaklık.
- Top-k.
- Seed.

## Model Çıktılarını Nasıl Yorumlamalı?

Bu proje büyük bir üretken yapay zeka sistemi değildir. Bilerek küçük tutulmuştur. Bu nedenle çıktılar şu şekilde yorumlanmalıdır:

- N-gram modeli küçük veri üzerinde daha okunur çıktılar verebilir, çünkü yerel karakter bağlamlarını ezberler.
- Transformer daha güçlü bir mimaridir, fakat mini veri kümesinde yeterince öğrenemez.
- Düşük perplexity her zaman iyi yaratıcılık anlamına gelmez; özellikle küçük veri kümesinde ezberleme belirtisi olabilir.
- Daha iyi hikaye üretimi için veri kümesi büyütülmelidir.

## Deney Fikirleri

- N-gram derecesi: 3, 5, 7 karşılaştırması.
- Sıcaklık ve top-k: `0.6`, `0.9`, `1.2` ile `top-k=4/8/16` değerlerinin akıcılığa etkisi.
- Veri boyutu: Mini derlem, Ömer Seyfettin derlemi ve daha geniş lisanslı Türkçe hikayelerle eğitim.
- Değerlendirme: Perplexity yanında insan değerlendirmesi, tekrar oranı ve Türkçe karakter hataları.
- Model ailesi: N-gram ile mini Transformer'ı aynı promptlar ve insan değerlendirmesiyle karşılaştırma.
- Prompt hassasiyeti: Eğitim verisinde geçen ve geçmeyen başlangıçların çıktı kalitesine etkisi.

## Araştırma Bildirisi İçin Başlık Önerisi

> Küçük Veri Kümesiyle Türkçe Hikaye Üretimi: Karakter Düzeyi N-gram ve Mini Transformer Karşılaştırması

Bu başlık altında proje şu noktaları tartışabilir:

- Türkçe karakter düzeyi modellemenin avantajları ve sınırları.
- Küçük veri kümesinde klasik olasılıksal modellerin davranışı.
- Transformer mimarisinin veri ihtiyacı.
- Perplexity ile insan değerlendirmesi arasındaki fark.
- Eğitim verisi boyutunun üretim kalitesine etkisi.

## Not

Bu derlem örnek amaçlıdır ve tamamıyla bu proje için yazılmış kısa metinlerden oluşur. Bildiri veya yayın için daha büyük ve lisansı açık bir veri kümesiyle deneyleri tekrarlamak gerekir.
