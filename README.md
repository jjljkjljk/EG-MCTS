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
Put all the files in ```dataset``` (```origin_dict.csv```,  ```uspto.reactions.json```, ```chembl.csv```, ```reaction_graph.gml```) under the ```eg_mcts/dataset``` directory.
Put the folder ```one_step_model/``` under the ```eg_mcts``` directory.
#### 3. Install EG-MCTS lib
    pip install -e eg_mcts/packages/mlp_retrosyn
    pip install -e eg_mcts/packages/rdchiral
    pip install -e .

#### 4. Data instruction

##### 1) Trian, vali and test sets
**For eMolecules,** ```emol_train_data.txt```, ```emol_vali_data.txt```,  ```emol_test_data.txt``` in ```eg_mcts/dataset```.
**For chembl,** ```chembl_train_data_T.txt```, ```chembl_train_data_T1.txt```,  ```chembl_train_data_T2.txt```, ```chembl_train_data_T^.txt```, ```chembl_train_data_T^^.txt```, ```cheml_vali_data.txt```, ```chembl_test_data.txt``` in ```eg_mcts/dataset```.
**For 30 test molecules used in EG-MCTS Versus Literature,** we provide their literature routes in SMILES format in ```eg_mcts/dataset/published_routes_for_30_mol```.

##### 2) Pre-trained model

In the folder ```eg_mcts/saved_EG_fn```, we provide the best model ```best_egn_for_emol.pt```  **for eMolecules.**
**For chembl,** , We provide the best models trained in different train data sets.

#### 5. Reproduce experiment results

##### 1) Planning on test set
To plan with EG_MCTS, run the following command,

    cd eg_mcts
    python EG_MCTS_plan.py




##### 2) Train your BEST EGN
Change the option ```--test_mols``` to your training molecule set, create a new random network and change the option ```--value_model``` to this new random network.

Then run the command

    cd eg_mcts
    python EG_MCTS_plan.py  --collect_expe

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
