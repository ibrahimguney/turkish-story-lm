# Turkce Hikaye Dil Modeli

Bu proje, Turkce hikayelerden sifirdan egitilen kucuk bir dil modeli laboratuvaridir. Amac iki yonludur:

1. Dil modeli egitiminin temel fikirlerini temiz ve okunabilir kodla ogrenmek.
2. Turkce hikaye uretimi uzerine bir arastirma bildirisi icin deney zemini kurmak.

Ilk surum karakter duzeyinde n-gram dil modeli kullanir. Bilerek bagimliliksizdir; Python standart kutuphanesiyle calisir. Bu sayede veri hazirlama, olasilik hesabi, kayip/perplexity ve metin uretimi adimlari seffaf bicimde incelenebilir.

Ikinci model, ayni veri uzerinde egitilen kucuk bir karakter duzeyi Transformer'dir. Bu model PyTorch gerektirir ve bildiride "klasik olasiliksal model" ile "sinir agi tabanli model" karsilastirmasi icin kullanilir.

## Kurulum

Python 3.10+ yeterlidir.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

Transformer deneyleri icin:

```bash
python -m pip install -e .[transformer]
```

Paket kurmadan da komutlari `python scripts/...` seklinde calistirabilirsiniz.

## Hizli Baslangic

Modeli mini Turkce hikaye derlemiyle egitin:

```bash
python scripts/train_ngram.py --data data/stories/tr_mini_stories.txt --out runs/ngram_tr.json --order 7 --alpha 0.001
```

Metin uretin:

```bash
python scripts/generate.py --model runs/ngram_tr.json --prompt Bir sabah --length 600 --temperature 0.7 --top-k 8 --out runs/ornek_hikaye.txt
```

Degerlendirme yapin:

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

## Proje Yapisi

- `data/stories/tr_mini_stories.txt`: Baslangic icin kisa, ozgun Turkce hikayeler.
- `src/turkish_story_lm/tokenizer.py`: Karakter duzeyi Turkce tokenizer.
- `src/turkish_story_lm/ngram.py`: Sifirdan yazilmis n-gram dil modeli.
- `src/turkish_story_lm/transformer.py`: Kucuk karakter duzeyi Transformer modeli.
- `scripts/train_ngram.py`: Egitim betigi.
- `scripts/generate.py`: Hikaye uretim betigi.
- `scripts/evaluate.py`: Ortalama negatif log olabilirlik ve perplexity.
- `scripts/train_transformer.py`: PyTorch Transformer egitim betigi.
- `scripts/generate_transformer.py`: Transformer ile hikaye uretimi.
- `scripts/evaluate_transformer.py`: Transformer icin kayip ve perplexity.
- `research/bildiri_zemini.md`: Arastirma bildirisi icin problem, deneyler ve taslak.

## Deney Fikirleri

- N-gram derecesi: 3, 5, 7 karsilastirmasi.
- Sicaklik ve top-k: `0.6`, `0.9`, `1.2` ile `top-k=4/8/16` degerlerinin akiciliga etkisi.
- Veri boyutu: Mini derlem yerine daha genis, lisansli Turkce hikayelerle egitim.
- Degerlendirme: Perplexity yaninda insan degerlendirmesi, tekrar orani ve Turkce karakter hatalari.
- Model ailesi: N-gram ile mini Transformer'i ayni promptlar ve insan degerlendirmesiyle karsilastirma.

## Not

Bu derlem ornek amaclidir ve tamamiyla bu proje icin yazilmis kisa metinlerden olusur. Bildiri veya yayin icin daha buyuk ve lisansi acik bir veri kumesiyle deneyleri tekrarlamak gerekir.
