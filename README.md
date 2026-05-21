# Turkce Hikaye Dil Modeli

Bu proje, Turkce hikayelerden sifirdan egitilen kucuk bir dil modeli laboratuvaridir. Amac iki yonludur:

1. Dil modeli egitiminin temel fikirlerini temiz ve okunabilir kodla ogrenmek.
2. Turkce hikaye uretimi uzerine bir arastirma bildirisi icin deney zemini kurmak.

Projede iki model vardir:

- **N-gram dil modeli**: Bagimliliksiz, seffaf ve hizli bir temel model. Karakterleri onceki karakterlere bakarak tahmin eder.
- **Mini Transformer**: PyTorch ile yazilmis kucuk bir sinir agi modeli. Ayni veri uzerinde n-gram yaklasimiyla karsilastirma yapmak icin eklenmistir.

Yerel web arayuzu sayesinde modelleri tarayicidan deneyebilir, prompt yazip hikaye uretimi yapabilirsiniz.

## Kurulum

Python 3.10+ yeterlidir.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

Transformer deneyleri icin PyTorch gerekir:

```bash
python -m pip install -e .[transformer]
```

Paket kurmadan da komutlari `python scripts/...` seklinde calistirabilirsiniz. Betikler kendi iclerinde `src/` klasorunu Python yoluna ekler.

## Hizli Baslangic

N-gram modelini mini Turkce hikaye derlemiyle egitin:

```bash
python scripts/train_ngram.py --data data/stories/tr_mini_stories.txt --out runs/ngram_tr.json --order 7 --alpha 0.001
```

N-gram ile metin uretin:

```bash
python scripts/generate.py --model runs/ngram_tr.json --prompt Bir sabah --length 600 --temperature 0.7 --top-k 8 --out runs/ornek_hikaye.txt
```

N-gram modelini degerlendirin:

```bash
python scripts/evaluate.py --model runs/ngram_tr.json --data data/stories/tr_mini_stories.txt
```

Transformer egitin:

```bash
python scripts/train_transformer.py --data data/stories/tr_mini_stories.txt --out runs/transformer_tr.pt --steps 500
```

Transformer ile metin uretin:

```bash
python scripts/generate_transformer.py --model runs/transformer_tr.pt --prompt Bir sabah --length 600 --temperature 0.8 --top-k 8 --out runs/transformer_ornek_hikaye.txt
```

Transformer'i degerlendirin:

```bash
python scripts/evaluate_transformer.py --model runs/transformer_tr.pt --data data/stories/tr_mini_stories.txt
```

Yerel web arayuzunu acin:

```bash
python scripts/serve_web.py --port 7860
```

Tarayicida `http://127.0.0.1:7860` adresine gidin.

Sunucu acikken hizli API testi:

```bash
python scripts/smoke_web.py
```

## Veri Akisi

Projedeki temel akis soyledir:

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
scripts/generate*.py, scripts/evaluate*.py veya web arayuzu
```

Veri dosyasi paragraflara ayrilmis kisa Turkce hikayelerden olusur. N-gram modelinde bu paragraflar ayri hikayeler olarak ele alinir. Transformer modelinde dosyanin tamamindaki karakter dizisi egitim verisi olarak kullanilir.

## Proje Yapisi

- `data/stories/tr_mini_stories.txt`: Baslangic icin kisa, ozgun Turkce hikayeler.
- `src/turkish_story_lm/tokenizer.py`: Karakter duzeyi tokenizer.
- `src/turkish_story_lm/ngram.py`: Sifirdan yazilmis n-gram dil modeli.
- `src/turkish_story_lm/transformer.py`: Kucuk karakter duzeyi Transformer modeli.
- `scripts/train_ngram.py`: N-gram egitim betigi.
- `scripts/generate.py`: N-gram hikaye uretim betigi.
- `scripts/evaluate.py`: N-gram icin ortalama negatif log olabilirlik ve perplexity.
- `scripts/train_transformer.py`: PyTorch Transformer egitim betigi.
- `scripts/generate_transformer.py`: Transformer ile hikaye uretimi.
- `scripts/evaluate_transformer.py`: Transformer icin kayip ve perplexity.
- `scripts/serve_web.py`: Yerel web arayuzu ve JSON API.
- `scripts/smoke_web.py`: Yerel web API icin hizli kontrol.
- `web/`: Tarayicida calisan hikaye uretim arayuzu.
- `research/bildiri_zemini.md`: Arastirma bildirisi icin problem, deneyler ve taslak.
- `research/deney_notlari.md`: Ilk deney sonuclari ve yorumlar.
- `runs/`: Egitilmis modeller ve uretilmis ornekler.

## Kodlar Nasil Calisiyor?

### `tokenizer.py`

Bu dosya karakter duzeyinde basit bir tokenizer tanimlar.

Temel fikir:

- `CharTokenizer.train(text)` metindeki benzersiz karakterleri bulur.
- `encode(text)` her karakteri sayisal kimlige cevirir.
- `decode(token_ids)` sayisal kimlikleri tekrar metne cevirir.
- Turkce karakterler ayri isleme tabi tutulmaz; Python stringleri Unicode destekledigi icin dogrudan korunur.

Bu proje sozcuk veya alt-sozcuk tokenizer kullanmaz. Bunun nedeni, dil modeli mantigini en sade haliyle gostermektir.

### `ngram.py`

Bu dosya karakter duzeyi n-gram dil modelini icerir.

N-gram modeli su soruyu cevaplar:

```text
Onceki n-1 karakteri gordugumde siradaki karakter ne olabilir?
```

Ornek:

```text
Baglam: "Bir saba"
Siradaki karakter adaylari: "h", "n", "r", ...
```

Modelin ana bolumleri:

- `fit(texts)`: Egitim metinlerinden baglam-karakter sayimlari cikarir.
- `next_token_distribution(context)`: Verilen baglam icin sonraki karakter olasiliklarini hesaplar.
- `generate(...)`: Olasiliklara gore karakter secerek metni uzatir.
- `negative_log_likelihood(text)`: Modelin bir metni ne kadar iyi tahmin ettigini olcer.
- `perplexity(text)`: Dil modellemede yaygin kullanilan okunabilirlik/olasilik metriğini verir.

Model her karakter icin yalnizca en uzun baglami degil, daha kisa baglamlari da kaydeder. Bu onemlidir; cunku kullanici egitim verisinde aynen gecmeyen bir prompt yazdiginda model tamamen rastgele secime dusmez. Uzun baglam bulunamazsa daha kisa baglamlarla tahmin yapar.

`alpha` parametresi additive smoothing icindir. Yani modelin az gorulen veya hic gorulmeyen karakterlere sifir olasilik vermesini engeller.

`top_k` parametresi uretim sirasinda sadece en olasi `k` karakter arasindan secim yapar. Bu, kucuk veriyle calisirken anlamsiz noktalama veya rastgele karakter cikma riskini azaltir.

### `transformer.py`

Bu dosya kucuk bir causal Transformer dil modeli tanimlar.

Modelin temel parcalari:

- `TransformerConfig`: Model boyutlarini tutan ayar sinifi.
- `TinyTransformerLM`: PyTorch `nn.Module` olarak yazilmis dil modeli.
- `token_embedding`: Karakter kimliklerini vektore cevirir.
- `position_embedding`: Karakterlerin siradaki konum bilgisini modele ekler.
- `TransformerEncoderLayer`: Self-attention ve feed-forward katmanlari.
- `head`: Son vektorleri karakter olasiliklarina ceviren lineer katman.

Bu model causal maske kullanir. Yani model, siradaki karakteri tahmin ederken gelecekteki karakterleri goremez. Bu davranis dil modeli egitimi icin gereklidir.

`generate(...)` fonksiyonu n-gram modelindeki gibi prompttan baslar, sonra karakterleri tek tek uretir. Sicaklik ve top-k ayarlari burada da kullanilir.

Mini veri kumesinde Transformer'in n-gramdan daha zayif gorunmesi normaldir. Transformer daha esnek bir modeldir, fakat iyi sonuc icin daha fazla veri ve daha uzun egitim ister.

## Betikler

### `scripts/train_ngram.py`

Bu betik n-gram modelini egitir.

Is akisi:

1. `--data` ile verilen hikaye dosyasini okur.
2. Bos satirlara gore hikayeleri ayirir.
3. `NGramLanguageModel(order, alpha)` olusturur.
4. `fit(stories)` ile karakter baglamlarini sayar.
5. Modeli `--out` konumuna JSON olarak kaydeder.

Cikti dosyasi insan tarafindan okunabilir JSON'dur. Bu, n-gram modelin nasil sayim yaptigini incelemek icin faydalidir.

### `scripts/generate.py`

Bu betik egitilmis n-gram modelinden hikaye uretir.

Onemli parametreler:

- `--prompt`: Hikayenin baslangic metni.
- `--length`: Uretilecek en fazla yeni karakter sayisi.
- `--temperature`: Dusuk deger daha guvenli, yuksek deger daha cesur uretim verir.
- `--top-k`: Her adimda en olasi kac karakter arasindan secim yapilacagi.
- `--seed`: Ayni ayarlarla tekrar edilebilir uretim saglar.
- `--out`: Sonucu dosyaya kaydeder.

### `scripts/evaluate.py`

Bu betik n-gram modelini veri dosyasi uzerinde degerlendirir.

Raporladigi degerler:

- `mean_nll`: Ortalama negatif log olabilirlik. Dusuk olmasi daha iyidir.
- `mean_perplexity`: Modelin tahmin belirsizligini ozetler. Dusuk olmasi daha iyidir.

Mini veri uzerinde cok dusuk perplexity ezberleme anlamina gelebilir. Bu nedenle metrikleri insan degerlendirmesiyle birlikte yorumlamak gerekir.

### `scripts/train_transformer.py`

Bu betik Transformer modelini egitir.

Is akisi:

1. Veri dosyasini okur.
2. `CharTokenizer` ile karakterleri sayisal kimliklere cevirir.
3. Rastgele mini-batch'ler olusturur.
4. Modelin siradaki karakteri tahmin etmesini ister.
5. Cross entropy loss ile hatayi hesaplar.
6. AdamW optimizer ile agirliklari gunceller.
7. Modeli `runs/transformer_tr.pt` dosyasina kaydeder.

Kaydedilen `.pt` dosyasi sunlari icerir:

- Model ayarlari.
- Tokenizer sozlugu.
- Model agirliklari.
- Egitim meta verisi.

### `scripts/generate_transformer.py`

Bu betik egitilmis Transformer modelinden metin uretir. `generate.py` ile benzer parametrelere sahiptir, fakat model dosyasi `.pt` formatindadir.

### `scripts/evaluate_transformer.py`

Bu betik Transformer modelini sabit uzunluklu parcalar uzerinden degerlendirir. Her parca icin siradaki karakter tahmin kaybi hesaplanir ve ortalamasi raporlanir.

### `scripts/serve_web.py`

Bu betik yerel web uygulamasini baslatir.

Iki is yapar:

- `web/` klasorundeki HTML, CSS ve JavaScript dosyalarini sunar.
- `/api/generate` ve `/api/models` endpoint'lerini saglar.

Endpointler:

```text
GET  /api/models
POST /api/generate
```

`POST /api/generate` su alanlari alir:

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

Sunucu acikken API'nin calisip calismadigini hizlica kontrol eder. Once modellerin mevcut olup olmadigini sorar, sonra kisa bir n-gram uretim istegi yollar.

## Web Arayuzu

`web/` klasoru tek sayfalik bir arayuz icerir.

- `web/index.html`: Sayfa iskeleti ve form alanlari.
- `web/styles.css`: Gorsel tasarim.
- `web/app.js`: Formdan verileri okur, API'ye istek atar ve sonucu ekrana yazar.

Arayuzde su ayarlar degistirilebilir:

- Prompt.
- Model secimi.
- Uretim uzunlugu.
- Sicaklik.
- Top-k.
- Seed.

## Model Ciktilarini Nasil Yorumlamali?

Bu proje buyuk bir uretken yapay zeka sistemi degildir. Bilerek kucuk tutulmustur. Bu nedenle ciktilar su sekilde yorumlanmalidir:

- N-gram modeli kucuk veri uzerinde daha okunur ciktilar verebilir, cunku yerel karakter baglamlarini ezberler.
- Transformer daha guclu bir mimaridir, fakat mini veri kumesinde yeterince ogrenemez.
- Dusuk perplexity her zaman iyi yaraticilik anlamina gelmez; ozellikle kucuk veri kumesinde ezberleme belirtisi olabilir.
- Daha iyi hikaye uretimi icin veri kumesi buyutulmalidir.

## Deney Fikirleri

- N-gram derecesi: 3, 5, 7 karsilastirmasi.
- Sicaklik ve top-k: `0.6`, `0.9`, `1.2` ile `top-k=4/8/16` degerlerinin akiciliga etkisi.
- Veri boyutu: Mini derlem yerine daha genis, lisansli Turkce hikayelerle egitim.
- Degerlendirme: Perplexity yaninda insan degerlendirmesi, tekrar orani ve Turkce karakter hatalari.
- Model ailesi: N-gram ile mini Transformer'i ayni promptlar ve insan degerlendirmesiyle karsilastirma.
- Prompt hassasiyeti: Egitim verisinde gecen ve gecmeyen baslangiclarin cikti kalitesine etkisi.

## Arastirma Bildirisi Icin Baslik Onerisi

> Kucuk Veri Kumesiyle Turkce Hikaye Uretimi: Karakter Duzeyi N-gram ve Mini Transformer Karsilastirmasi

Bu baslik altinda proje su noktalari tartisabilir:

- Turkce karakter duzeyi modellemenin avantajlari ve sinirlari.
- Kucuk veri kumesinde klasik olasiliksal modellerin davranisi.
- Transformer mimarisinin veri ihtiyaci.
- Perplexity ile insan degerlendirmesi arasindaki fark.
- Egitim verisi boyutunun uretim kalitesine etkisi.

## Not

Bu derlem ornek amaclidir ve tamamiyla bu proje icin yazilmis kisa metinlerden olusur. Bildiri veya yayin icin daha buyuk ve lisansi acik bir veri kumesiyle deneyleri tekrarlamak gerekir.
