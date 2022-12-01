# APAN: Asynchronous Propagation Attention Network for Real-time Temporal Graph Embedding

## Key words:
Graph Neural Network, Graph Embedding, Continuous-Time Dynamic Graph, Link Perdiction

## Requirements:
pytorch>=1.5.0 DGL==0.5.2 numpy>=1.19 python >= 3.6

## Dataset and Preprocessing
### Download the public data
Download the datasets (eg. wikipedia and reddit) from [here](http://snap.stanford.edu/jodie/#datasets) and store their csv files in a folder named data/.
Note that the csv file should be renamed as wikipedia_raw.csv or reddit_raw.csv 

### Preprocess the data
Step 1. python preprocess/preprocess_csv.py --data wikipedia/reddit

Step 2. python preprocess/BuildDglGraph.py --data wikipedia/reddit


## Model Training
python train.py -d wikipedia/reddit

### General flags
Optional arguments are described in utils/args.py.

## TODO

## Cite us
Will be appeared at SIGMOD 2021
