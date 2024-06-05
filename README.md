# ReAuKGC: Relation-Augmented for Knowledge Graph Completion with Pre-trained Language Model


Source code for the paper: ReAuKGC:Relation-Augmented for Knowledge Graph Completion with Pre-trained Language Model 


## Requirements

- [PyTorch](http://pytorch.org/) version >= 1.7.1
- [NumPy](http://numpy.org/) version >= 1.19.5
- transformers
- tqdm
- Python version >= 3.6

## Usage

Run main.py to train or test our models. 


## How to Run

```bash
python main.py --batch_size 16 --epoch 100 --plm bert  --data wn18rr --top_m 6
```
```bash
python main.py --batch_size 16 --epoch 10 --plm bert  --data fb15k-237 --top_m 20
```


The arguments are as following:
* `--bert_lr`: learning rate of the language model.
* `--model_lr`: learning rate of other parameters.
* `--batch_size`: batch size used in training.
* `--weight_decay`: weight dacay used in training.
* `--data`: name of the dataset. Choose from 'fb15k-237', 'wn18rr', 'fb13' and 'umls'.
* `--plm`: choice of the language model. Choose from 'bert' and 'bert_tiny'.
* `--load_path`: path of checkpoint to load.
* `--load_epoch`: load the checkpoint of a specific epoch. Use with --load_metric.
* `--load_metric`: use with --load_epoch.
* `--top_m`: the top m that need to be re-ranking.



### Datasets

The datasets are put in the folder 'data', including fb15k-237, WN18RR, FB13 and umls.


