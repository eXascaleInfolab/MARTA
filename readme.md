# MARTA: Leveraging Human Rationales for Explainable Text Classification
MARTA is a unified Bayesian Framework that integrates an attention-based model with labels and rationales contributed by workers.
## Structure of MARTA's repository
- This repo is composed of four main directories:
	- code: source code of MARTA
	- data: contains two subfolders which are "original data" with the two datasets Amazon and Wiki_tech used in the paper, and "processed_data" which contains the generated files from the original data needed to run MARTA.
	- scripts: contains two scripts to run the code on the datasets used in the paper
	- results: after running the code, the results will be saved in this directory

## Using MARTA
### Create a virtual environment and install requirements
We are using python3.6 on a Ubuntu 16.04 machine with 32 CPUs and 128GB RAM. You can create a virtual environment for MARTA using the following command:
``` bash
sudo apt-get install python3-venv
python3.6 -m venv env-marta
source env-marta/bin/activate
```
Install all requirements using the following command:
``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Running MARTA
To run MARTA on the amazon data, you can use the script: amazon.sh in the scripts folder
``` bash
chmod u+x ./scripts/amazon.sh
cd code
../scripts/amazon.sh
```
To run MARTA on the wiki_tech data, you can use the script: wiki_tech.sh in the scripts folder
``` bash
chmod u+x ./scripts/wiki_tech.sh
cd code
../scripts/wiki_tech.sh
```

## Using MARTA with other datasets:
- In order to generate the data needed for MARTA, you can use the python script gen_data_marta.py.
- The script takes as input a csv file with the following header:
doc_id,text,label,WorkerId,worker_label,rationale, where:
	- doc_id: the document id
	- text: content of the document in one line
	- label: ground truth binary label of the document
	- WorkerId: the worker id
	- worker_label: the label given by the worker
	- rationale: the part of text selected by the worker as a justification to her label
The scripts generates three files:
	- textual_data.csv: contains for each document, the sentences composing it and the ground truth label of the document. The format of the generated file is 'doc_id', 'text','sentence','label'.
	- workers_answers.csv: contains the worker labeling. The format of the file is 'doc_id','WorkerId','worker_label'
	- workers_sentence_label.csv: contains worker labeling at the sentence level. The format of the file is 'doc_id','WorkerId','worker_label','sentence','rationale','sentence_label'

Examples of running the code:
``` bash
cd ./code/data_process/
```
- For Amazon data:
``` bash
python gen_data_marta.py --original_data '../../data/original_data/amazon.csv' --dir_gen_marta '../../data/processed_data/amazon/'
```

## Citation

Please cite the following paper when using MARTA:
``` bash
@inproceedings{arous2021aaai,
  title = {MARTA: Leveraging Human Rationales for Explainable Text Classification},
  author = {Arous, Ines and Dolamic, Ljiljana and Yang, Jie and Bhardwaj, Akansha and Cuccu, Giuseppe and Cudr{\'e}-Mauroux, Philippe},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2021)},
  year = {2021},
  address = {A Virtual Conference}
}
```
