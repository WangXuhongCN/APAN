# The Pytorch and DGL implement of the paper.

## Requirements:
pytorch>=1.5.0 DGL>=0.5.2 numpy>=1.19 python >= 3.6

## Dataset and Preprocessing
### Download the public data
Download the sample datasets (eg. wikipedia and reddit) from [here](http://snap.stanford.edu/jodie/#datasets) and store their csv files in a folder named data/.

### Preprocess the data
Step 1. python preprocess/preprocess_csv.py --data wikipedia/reddit

Step 2. python preprocess/BuildDglGraph.py --data wikipedia/reddit


## Model Training
python train.py -d wikipedia/reddit

### General flags
Optional arguments are described in utils/args.py.

## TODO

## Cite us
