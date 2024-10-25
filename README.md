## About
Chemical safety assessment is critical in the early stages of drug discovery and ecological risk assessment. Multi-species acute toxicity assessment is typically conducted as the initial phase to determine whether chemicals can proceed to industrial use or clinical trials. Although deep learning has shown promise in acute toxicity evaluation, inherent challenges such as diverse experimental conditions, imbalanced endpoint data, and scarce target endpoint data are often overlooked. This hinders existing methods from revealing associations among multi-condition endpoints, and leads to poor predictive performance for data-scarce target endpoints, especially those related to humans. Here we propose a novel machine learning paradigm, Adjoint Correlation Learning, for multi-condition acute toxicity assessment (ToxACoL) to address these challenges. 

**ToxACoL**  models biological associations among multi-species, multi-condition toxicity endpoints via graph topology and achieves knowledge transfer via graph convolution. An adjoint correlation mechanism encodes compounds and endpoints synchronously, enabling an endpoint-aware and task-focused representation learning for compounds. Comprehensive analyses demonstrate that ToxACoL successfully balances performance across multi-condition endpoints, yielding 43%-115% improvements for data-scarce human-related endpoints, while reducing required training data by approximately 70% to 80%. Furthermore, investigation into the visualization and interpretability of the top-level representation learned by ToxACoL elucidates the structural alert mechanisms behind acute toxicity and highlights the potential for extrapolating animal test results to humans when integrated with the filled-in toxicity values.

This git repository contains the main source codes of ToxACoL.

## Code Environment
Please refer to the `requirments.txt`. Some main tools or libraries and their versions were listed below:
```
python==3.10.12
torch==2.2.1
torchnet==0.0.4
pandas==2.2.1
```

## Installation
```sh
git clone https://github.com/LuJiangTHU/Acute_Toxicity_FSL.git
cd Acute_Toxicity_FSL
```
Due to the large size of the molecular fingerprint file of compounds in the acute toxicity dataset, it cannot be directly uploaded to this git repository. So please download the `all_descriptors.txt` file from the following website (https://doi.org/10.6084/m9.figshare.27195339.v4) and place it in the directory of `./data/`

## Data Description
The acute toxicity dataset includes 59 various toxicity endpoints with 80,081 unique compounds represented using SMILES strings, and 122,594 usable toxicity measurements described by continuous values with a unified toxicity chemical unit: -log(mol/kg). The larger the measurement value, the stronger the toxicity intensity of the corresponding compound towards a certain endpoint. The 59 acute toxicity endpoints involve 15 different species including mouse, rat, rabbit, guinea pig, dog, cat, bird wild, quail, duck, chicken, frog, mammal, man, women, and human, 8 different administration routes including intraperitoneal, intravenous, oral, skin, subcutaneous, intramuscular, parenteral, and unreported, and 3 different measurement indicators including LD50 (lethal dose 50%), LDLo (lethal dose low), and TDLo (toxic dose low). In this dataset, each compound only has toxicity measurement values concerning a small number of toxicity endpoints, so this dataset is very sparse with nearly 97.4% of compound-to-endpoint measurements missing. Meanwhile, this dataset is also extremely data-unbalanced with some endpoints having tens of thousands of toxicity measurements available, e.g., mouse-intraperitoneal-LD50 has 36,295 measurements, mouse-oral-LD50 has 23,373 measurements, and rat-oral-LD50 has 10,190 measurements, etc, while some endpoints contain only around 100 measurements like mouse-intravenous-LDLo, rat-intravenous-LDLo, frog-subcutaneous-LD50, and human-oral-TDLo, etc. The sparsity and unbalance of this dataset present acute toxicity evaluation as a challenging issue. Among the 59 endpoints, 21 endpoints with less than 200 measurements were considered small-sized endpoints, and 11 endpoints with more than 1000 measurements were treated as large-sized endpoints. Three endpoints targeting humans, human-oral-TDLo, women-oral-TDLo, and man-oral-TDLo, are typical small-sized endpoints, with only 140, 156, and 163 available toxicity measurements, respectively 

The acute toxicity intensity measurement values of the 80,081 compounds concerning 59 acute toxic endpoints, as well as the 5-fold random splits, were provided in the `./data/dataset.txt`. 

The molecular fingerprints or feature descripors of the 80,081 compounds, such as Avalon, Morgan, and AtomPair, were given in the `./data/all_descriptors.txt`.

The acute toxicity dataset was randomly divided 5 times. The specific data splits were provided in the folder `./data/random split/`, which were strictly consistent with the other baseline methods to ensure comparison fairness.

## Training
The `./config/cfg_ToxACoL_xx.py` can be used to control the training or testing configurations including the network architecture, train/test fold files, type of molecular fingerprints, optimizer parameters, and random seeds, etc. 

In this repository, we gave 5 examples of cfg files, `cfg_ToxACoL_4layer_f0.py`, `cfg_ToxACoL_4layer_f1.py`, `cfg_ToxACoL_4layer_f2.py`, `cfg_ToxACoL_4layer_f3.py`, and `cfg_ToxACoL_4layer_f4.py`. The 4-layer ToxACoL is the main model we adopted in this study. `f0-f4` represent the 5-fold cross-validation experiments.

Using the following command to train your ToxACoL on fold0:
```sh
python train_multimodel.py --config cfg_ToxACoL_4layer_f0 --num_workers 8
```
Correspondingly, the logs produced from experiments, snapshots and the optimal model files will saved into `./experiments/cfg_ToxACoL_4layer_f0`.


## Evaluating
After trained using 5 cross-validation folds, you can use the `consensus_evaluation.py` to evaluate the final averaged  performance on 5 cross-validation folds:
```sh
python consensus_evaluation.py
```
The results will be saved as the form of tables and placed under the folder  `./table_results/`. 


