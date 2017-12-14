# duke -- Dataset Understanding via Knowledge-graph Entities

First, you should download wiki2vec model (English Wikipedia) using a torrent as described on this link:

https://github.com/idio/wiki2vec

Untar it in the top-level folder of this project

Second, got to https://datadrivendiscovery.org/data/seed_datasets_current/, download the dataset you want and put it in the darpa/data/seeds subfolder

Change macro settings on Ln 72 to 78 of main.py according to what you want to ananalyze (name of dataset, column name, look at headers instead? etc.). The code should be documented sufficiently to be easily understood.

Third, make sure to use python 2.7 (for now) and be sure to get gensim installed via `pip2 install gensim`

Finally, run it as follows, and enjoy the magic... 

```bash
python2 main.py
```

