# NMPA
Tensorflow implementation of Learning Non-myopic Power Allocation in Constrained Scenarios (https://arxiv.org/abs/)

## Overview
This library contains a Tensorflow implementation of Learning Non-myopic Power Allocation in Constrained Scenarios as presented in [[1]](#citation)(https://arxiv.org/abs/).

## Dependencies

* **python>=3.6**
* **tensorflow>=2.0**: https://tensorflow.org
* **tensorflow_addons**
* **numpy**
* **matplotlib**

## Structure
* [main](https://github.com/ArCho48/UWMMSE-MIMO/blob/master/train.py): Main code for running the experiments in the paper. Run as python3 train.py 
* [model](https://github.com/ArCho48/UWMMSE-MIMO/blob/master/model.py): Defines the UWMMSE model.
* [data](https://github.com/ArCho48/UWMMSE-MIMO/tree/master/data): Should contain your dataset in folder {dataset ID}. 
* [models](https://github.com/ArCho48/UWMMSE-MIMO/tree/master/models): Stores trained models in a folder with same name as {datset ID}.
* [results](https://github.com/ArCho48/UWMMSE-MIMO/tree/master/results): Stores results in a folder with same name as {datset ID}.

## Usage


Please cite [[1](#citation)] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [Arindam Chowdhury](mailto:arindam.chowdhury@rice.edu).

## Citation
```
[1] Chowdhury A, Verma G, Swami A, Segarra S. Deep Graph Unfolding for Beamforming in MU-MIMO Interference Networks. 
arXiv preprint arXiv:.
```

BibTeX format:
```
@article{chowdhury2023deep,
  title={Learning Non-myopic Power Allocation in Constrained Scenarios},
  author={Chowdhury, Arindam and Paternain, Santiago and Verma, Gunjan and Swami, Ananthram and Segarra, Santiago},
  journal={arXiv e-prints},
  year={2023}
}
```
