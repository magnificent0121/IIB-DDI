# Improving efficiency in rationale discovery for Out-of-Distribution molecular representations

### Requirements
- Python version: 3.7.10
- Pytorch version: 1.8.1
- torch-geometric version: 1.7.0


### Download and Create datasets
- Download Drug-Drug Interaction dataset from https://github.com/isjakewong/MIRACLE/tree/main/MIRACLE/datachem.
    - Since these datasets include duplicate instances in train/validation/test split, merge the train/validation/test dataset.
    - Generate random negative counterparts by sampling a complement set of positive drug pairs as negatives.
    - Split the dataset into 6:2:2 ratio, and create separate csv file for each train/validation/test splits.
- Put each datasets into ``data/raw`` and run ``data.py`` file.
- Then, the python file will create ``{}.pt`` file in ``data/processed``.

### Hyperparameters
Following Options can be passed to `main.py`

`--dataset:`
Name of the dataset. Supported names are: ZhangDDI, ChemDDI and DeepDDI.
usage example :`--dataset ZhangDDI`

`--lr:`
Learning rate for training the model.  
usage example :`--lr 0.001`

`--epochs:`
Number of epochs for training the model.  
usage example :`--epochs 500`

`--beta:`
Hyperparameters for balance the trade-off between prediction and compression.  
usage example :`--beta 1.0`

`--gamma`:
Hyperparameter to control the weight of the VQ loss in the total loss.  
usage example: `--gamma 0.5`

`--vq_beta`:
Hyperparameter to control the weight of the commitment loss in the VQ loss.  
usage example: `--vq_beta 1`

`--vq_delta`:
Hyperparameter to control the weight of the embedding loss in the VQ loss.  
usage example: `--vq_delta 0.25`

Note: alpha is the ratio of hyperparameters vq_beta and vq_delta.

`--vq_num_embeddings`:
Hyperparameter to control the number of embeddings in the VQ loss.  
usage example: `--vq_num_embeddings 512`

`--sample_times`:
Hyperparameter to control the number of samples for environment sampling.  
usage example: `--sample_times 3`
