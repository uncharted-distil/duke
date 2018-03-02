# duke -- Dataset Understanding via Knowledge-graph Embeddings

First, you should download wiki2vec model (English Wikipedia) using a torrent as described on this link:

https://github.com/idio/wiki2vec

Untar it in the top-level folder of this project

Second, go to https://datadrivendiscovery.org/data/seed_datasets_current/, download the dataset you want and put it in the darpa_data/seeds subfolder

Third, make sure to use python 3.5+ and be sure to get gensim installed via `pip3 install gensim`

Finally, run it as follows, and enjoy the magic... 

```bash
python3 main.py
```

