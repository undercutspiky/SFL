# Shape-based Feature Learning
Official repo for NeurIPS 2024 paper "Are nuclear masks all you need for improved out-of-domain generalization? A closer look at cancer classification in histopathology"

Watch my presentation on YouTube to get an overview of the method, the motivation, main results and ablation studies: https://www.youtube.com/watch?v=zalQrK5p7x8

## Quick start:
1. Install the required packages listed in requirements.txt using `pip install -r requirements.txt`.
2. Download the processed CAMELYON17 dataset from https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/NXPLFL into your data directory.
3. Run `python train.py --config_path configs/config_sfl.py --hospital 3 --data_dir <CAMELYON17_DATA_DIR_PATH>` to train a model on hospital-3 (center-3) data using our method. Replace the config path for training a baseline model. You should modify the checkpoints directory path and the model name in the code. If you're going to train multiple models per hospital then you MUST modify the model name to prevent the code from overwriting the checkpoint path for that hospital. In my original code, I appended the `<SLURM JOB ID>` to the model name.
4. To test the models, run `python test.py --ckpts_path <PATH_TO_CEHCKPOINTS_DIRECTORY> --data_dir <CAMELYON17_DATA_DIR_PATH>`.

### Overview of directories:

**configs:** Contains configurations used for our method (SFL) and the baseline method.

**datasets:** Contains code for reading datasets as PyTorch datasets.

**preprocess:** Contains subdirectories, one for each dataset, processing the original raw datasets. While I have uploaded the processed CAMELYON17 dataset on dataverse.no (see the link above), you'd need to download and process the BCSS and OCELOT datasets yourself (using the code in the corresponding subdirectory) if you'd like to use them as test sets.

### Hyperparameters / settings specific to our method (SFL):
The 3 main hyperparameters (or setup) that you'd need to tune for our method are a) the hyperparameter lambda in the paper, b) when to introduce the l2-distance loss in the training, and c) when to start saving the model. b) and c) should be fairly easy to tune while the actual hyperparameter lambda will require some experimentation.

**When to introduce l2-distance loss in training:** For CAMELYON17 dataset, introducing the loss from the beginning also works but for one or two hospitals the network took tons of epochs to get out of the collapsed state where it was predicting the same class for all the samples and had the same loss for every step. So, it's probably better to train it for a few epochs (pick a random low number) before you introduce the l2-distance.

**When to start saving the model:** Once you introduce the l2-distance, the network will get destabilised for a few epochs and the total loss on the validation set might hit a very low number when the network's embeddings are close for 0 for all the samples thus making the distance loss very small. But you should see an inflection point on the loss graph where  the loss on the validation set starts going up again as the network actually starts learning something useful. You can start saving the model after the network has passed that inflection point...basically when it has stabilised.

**Hyperparameter lambda:** This totally depends on your dataset. For CAMELYON17, simply setting it to 1 works perfectly well and I suspect that it will work for any dataset that is big (and somewhat diverse as in has more than 1 patient) enough. So, start with the default value 1 but try reducing it using binary search with 1e-5 (anything above 0) as the min value and 1 as the max value.