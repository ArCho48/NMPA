# NMPA
Tensorflow implementation of Learning Non-myopic Power Allocation in Constrained Scenarios (https://arxiv.org/abs/)

## Overview
This library contains a Tensorflow implementation of Learning Non-myopic Power Allocation in Constrained Scenarios as presented in [[1]](#citation)(https://arxiv.org/abs/).

## Dependencies

* **python>=3.6**
* **tensorflow>=2.0**: https://tensorflow.org
* **numpy**
* **matplotlib**

## Structure
* [train](https://github.com/ArCho48/NMPA/blob/master/train.py): Code for training the NMPA model. Run as python3 train.py  --set {expID} with default parameters.
* [model](https://github.com/ArCho48/NMPA/blob/master/model.py): Defines the NMPA model.
* [run](https://github.com/ArCho48/NMPA/blob/master/run.py): Code for running the trained NMPA model. Run as python3 run.py  --set {expID} with default parameters.
* [train_uwmmmse](https://github.com/ArCho48/NMPA/blob/master/train_uwmmse.py): Code for training the lower-level UWMMSE model. Run as python3 train_uwmmse.py with default parameters.
* [data](https://github.com/ArCho48/NMPA/tree/master/data): Should contain your dataset in folder {expID}.
* [models](https://github.com/ArCho48/NMPA/tree/master/models): Stores pretrained models in a folder with same name as {expID}.
* [checkpoints](https://github.com/ArCho48/NMPA/tree/master/checkpoints): Stores trained models in a folder with same name as DDPG/{expID}.
* [results](https://github.com/ArCho48/NMPA/tree/master/results): Stores results in a folder with same name as {datset ID}.

Please cite [[1](#citation)] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [Arindam Chowdhury](mailto:arindam.chowdhury@rice.edu).

## Citation
```
[1] Chowdhury A, Paternain S, Verma G, Swami A, Segarra S. Learning Non-myopic Power Allocation in Constrained Scenarios. 
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
