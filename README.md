# Retrosynthetic Planning with EG-MCTS
Retrosynthetic Planning With Experience-Guided Monte Carlo Tree Search

#### 1. Setup the environment
##### 1) Download the repository
    cd EG-MCTS
##### 2) Create a conda environment
    conda env create -f environment.yml
    conda activate eg_mcts_env

#### 2. Download the data
##### 1) Download the building block set, pretrained one-step retrosynthetic model, and (optional) reactions from USPTO.
Download the files from this [link](https://drive.google.com/drive/folders/1fu70cs0-POpMpsPzbqIfkpdaQa24Iqk2?usp=sharing).
Put all the files in ```dataset``` (```origin_dict.csv```,  ```uspto.reactions.json```) under the ```eg_mcts/dataset``` directory.
Put the folder ```one_step_model/``` under the ```eg_mcts``` directory.
#### 3. Install EG-MCTS lib
    pip install -e eg_mcts/packages/mlp_retrosyn
    pip install -e eg_mcts/packages/rdchiral
    pip install -e .

#### 4. Data instruction

##### 1) Test sets

The test set of 180 molecules for testing is in ```eg_mcts/dataset```.


##### 2) Pre-trained model

In the folder ```eg_mcts/saved_EG_fn```, we provide the best model ```best_EGN.pt``` which you can use it to reproduce the experiments.

#### 5. Reproduce experiment results

##### 1) Planning on test set
To plan with EG_MCTS, run the following command,

    cd eg_mcts
    python EG_MCTS_plan.py --use_value_fn

Ignore the ```--use_value_fn``` option to plan without the EGN.
##### 2) Train a new EGN one round
You can also train a new EGN via,

    python train.py

then get a EGN trained on your own experience datasets for 20 epochs.


##### 3) Train your BEST EGN
Change the option ```--test_mols``` to your training molecule set, create a new random network and change the option ```--value_model``` to this new random network.

Then run the command

    cd eg_mcts
    python EG_MCTS_plan.py --use_value_fn --collect_expe

to collect the synthetic experience of your training molecules, which will be saved in the file ```experience_dataset/train_experience.pkl```. (Or you can change it to your own file.)

And then train this new random network.

Change the option ```--train_data``` to your own synthetic experience file.

And then run the command,

    python train.py

You will get a new EGN trained for one round.

If you want to train it for multiple time, then repeat the process above.

Don't forget to change the related options.
#### 5. Example usage

See ```example.py``` for an example usage.
