Usage:
-
0. Install python3 via `apt-get` or `brew`
1. Install additional libraries via `pip3 install -r requirements.txt`
2. In order to use preprocessed data run `python3 prepare.py`. The script will download ~1.2Gb of data. If you want to download glove embeddings and generate essential data run `python3 prepare.py --preprocess yes`. Warning: estimated time ~1 hour (time consuming lemmatization). `--embeddings` arg allows to provide path to unarchived and [downloaded](https://nlp.stanford.edu/projects/glove/) embeddings (840B, 300d).
3 (Optional). Train model via `python3 train.py`. In order to change some hyperparameters refer to help: `python3 train.py -h`.
4. In order to evaluate F1 score on test dataset, use `python3 test.py`
5. In order to turn on an interactive mode, use `python3 demo.py`


System requirements:
- 
- External GPU (NVIDIA K80 or similar)
- 8+ Gb of RAM
- Multicore CPU (recommended)
- Ubuntu 16.04 (or higher)
