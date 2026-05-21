# Deney Notlari

Bu dosya, ilk calistirmalardan elde edilen kisa gozlemleri kaydetmek icin tutulur.

## 1. N-gram Baslangic Modeli

Komut:

```bash
python scripts/train_ngram.py --data data/stories/tr_mini_stories.txt --out runs/ngram_tr.json --order 7 --alpha 0.001
python scripts/evaluate.py --model runs/ngram_tr.json --data data/stories/tr_mini_stories.txt
```

Sonuc:

- `stories=8`
- `mean_nll=0.1362`
- `mean_perplexity=1.1461`

Gozlem: Mini derlem uzerinde n-gram model hizla ezberlemeye yaklasiyor. Uretimler yer yer egitim cumlelerini kolajliyor; bu nedenle dusuk perplexity tek basina iyi genelleme gostergesi degil.

## 2. Transformer Baslangic Modeli

Komut:

```bash
python scripts/train_transformer.py --data data/stories/tr_mini_stories.txt --out runs/transformer_tr.pt --steps 500 --eval-interval 100 --eval-iters 5 --batch-size 8 --block-size 96 --n-embd 64 --n-head 4 --n-layer 2
python scripts/evaluate_transformer.py --model runs/transformer_tr.pt --data data/stories/tr_mini_stories.txt
```

Egitim izi:

- `step=1 train_loss=4.0405 eval_loss=4.0143`
- `step=100 train_loss=2.6963 eval_loss=2.6566`
- `step=200 train_loss=2.5107 eval_loss=2.4724`
- `step=300 train_loss=2.3673 eval_loss=2.3755`
- `step=400 train_loss=2.3544 eval_loss=2.3673`
- `step=500 train_loss=2.3393 eval_loss=2.2904`

Sonuc:

- `chunks=36`
- `mean_nll=2.3169`
- `mean_perplexity=10.1438`

Gozlem: Transformer egitim kaybini dusuruyor fakat mini veri kumesinde uretim henuz akici degil. Bu durum, daha genis veri kumesi, daha uzun egitim ve daha sistematik validation bolumu ihtiyacini gosteriyor.

## Ilk Yorum

N-gram kucuk veri uzerinde daha okunur cikti verebiliyor, cunku yerel baglamlari dogrudan ezberliyor. Transformer ise daha esnek bir mimari olmasina ragmen, bu veri olceginde yeterli istatistiksel sinyal alamiyor. Bildiri icin bu fark, veri olcegi ve model kapasitesi iliskisini tartismak acisindan kullanisli bir baslangic bulgusudur.
