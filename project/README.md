Usage:
-
0. Install python3 via `apt-get` or `brew`
1. Install additional libraries via `pip3 install -r requirements.txt`
2. (Optionally, recommended) Download pretrained glove embeddings 840B, 300d [here](https://nlp.stanford.edu/projects/glove/) and unzip archiv.
3. Run preprocessing via `python3 prepare.py --embeddings [path to glove]` Warning: estimated time ~1 hour (time consuming lemmatization). The script will create `extra_data` directory and store there all essential data. If not provided path to glove embeddings, the script will download the latter.
4. Train model via `python3 train.py`. In order to change some hyperparameters refer to help: `python3 train.py -h`.

- In order to evaluate F1 score on test dataset, use `python3 test.py`
- In order to turn on an interactive mode, use `python3 demo.py`

System requirements:
- 
- External GPU (NVIDIA K80 or similar)
- 8+ Gb of RAM
- Multicore CPU (recommended)
