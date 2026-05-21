# Turkce Hikaye Dil Modellemesi Icin Bildiri Zemini

## Calisma Amaci

Bu calisma, sinirli kaynakla Turkce hikaye uretimi yapabilen kucuk dil modellerinin ogretici ve arastirmaya uygun bir deney ortami olarak nasil kurulabilecegini inceler. Ilk asama, karakter duzeyinde n-gram dil modeliyle Turkce karakter dagilimi, yerel baglam ve kisa hikaye biceminin modellenmesini hedefler.

## Arastirma Sorulari

1. Karakter duzeyinde n-gram modeller Turkce hikaye bicemini ne kadar yakalayabilir?
2. N-gram derecesi arttikca akicilik, tekrar ve ezberleme davranisi nasil degisir?
3. Kucuk veri kumesinde olasiliksal model ile mini Transformer arasindaki farklar nelerdir?
4. Turkceye ozgu karakterler ve eklemeli yapi, karakter duzeyi modelde hangi hata kaliplarini uretir?

## Yontem

Baslangic modeli karakter duzeyinde n-gramdir. Model, her karakteri onceki `n-1` karaktere kosullu olarak tahmin eder. Gorulmeyen baglamlar icin kademeli backoff, dusuk frekansli olaylar icin additive smoothing kullanilir.

Karsilastirma modeli karakter duzeyinde kucuk bir Transformer'dir. Model token embedding, position embedding, nedensel self-attention ve feed-forward katmanlarindan olusur. Bu model ayni veri kumesiyle egitilerek n-gram yaklasiminin yerel baglam ezberi ile Transformer'in daha esnek temsil kapasitesi karsilastirilir.

Deneylerde ayni veri bolumu uzerinde farkli `n` degerleri denenebilir:

- `n=3`: Daha cesur ama kopuk uretimler.
- `n=5`: Mini veri icin dengeli baslangic.
- `n=7`: Daha yerel tutarli fakat ezberlemeye daha yatkin.

Transformer deneylerinde su degiskenler izlenebilir:

- Katman sayisi: `n_layer=1/2/4`.
- Gomme boyutu: `n_embd=64/128/256`.
- Baglam uzunlugu: `block_size=64/96/128`.
- Egitim adimi: `steps=300/1000/3000`.

## Metrikler

- Ortalama negatif log olabilirlik.
- Perplexity.
- Egitim kaybi ve validasyon kaybi.
- Model boyutu ve egitim suresi.
- Tekrar orani.
- Turkce karakter koruma orani.
- Insan degerlendirmesi: akicilik, anlamsal tutarlilik, hikaye hissi.

## Deney Plani

1. Mini derlemle n-gram modelini egit.
2. `n=3`, `n=5`, `n=7` icin ayni promptlardan ornekler uret.
3. Sicaklik degerlerini karsilastir: `0.6`, `0.9`, `1.2`.
4. Top-k orneklemeyi karsilastir: `top-k=4`, `top-k=8`, `top-k=16`.
5. Mini Transformer'i ayni veriyle egit ve ayni promptlarla ornekler uret.
6. N-gram ve Transformer ciktilarini metrikler ve insan degerlendirmesiyle karsilastir.
7. Uretimlerde tekrar, kopukluk, karakter hatasi ve anlatim tutarliligini etiketle.
8. Veri kumesi buyutuldugunde metriklerin nasil degistigini raporla.

## Beklenen Katki

Bu proje buyuk modellerle rekabet etmekten cok, Turkce dil modellemesi icin acik, anlasilir ve genisletilebilir bir egitim zemini sunar. Kodun bagimliliksiz olmasi, yontemin sinif icinde ve arastirma prototiplerinde kolayca incelenmesini saglar.

## Sonraki Asama

Bir sonraki teknik adim, mini derlemi lisansli ve daha genis Turkce hikayelerle buyutmek, ardindan n-gram ve Transformer deneylerini sabit prompt listesiyle tekrarlamaktir.
