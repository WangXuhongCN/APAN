# The Pytorch and DGL implement of the paper.

## Abstract
Limited by the time complexity of querying k-hop neighbors in a graph database, most graph algorithms cannot be deployed online and execute millisecond-level inference. This problem dramatically limits the potential of applying graph algorithms in certain areas, such as financial fraud detection. Therefore, we propose Asynchronous Propagate Attention Network, an asynchronous continuous time dynamic graph algorithm for real-time temporal graph embedding. Traditional graph models usually execute two serial operations: first graph computation and then model inference. We decouple model inference and graph computation step so that the heavy graph query operations will not damage the speed of model inference. Extensive experiments demonstrate that the proposed method can achieve competitive performance and 8.7 times inference speed improvement in the meantime. The source code is published at a Github repository.

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
