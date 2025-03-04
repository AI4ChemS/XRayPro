![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Open in Streamlit](https://img.shields.io/badge/Streamlit-Open%20in%20Streamlit-brightgreen)](https://xraypro.streamlit.app/)

## XRayPro

A recommendation system for MOFs leveraging only PXRDs and precursors. For further details, please refer to our [paper](https://chemrxiv.org/engage/chemrxiv/article-details/671a9d9783f22e42140f2df6). 

## Usage
A demo of the finetuning can be found in ```src/finetuning.ipynb```, in which the model is finetuned to predict methane uptake at HP on BW20K database. For pretraining purposes, please refer to ```src/pretrain.py```. For making predictions with loaded weights, please refer to ```src/predictions.ipynb```. For preprocessing experimental PXRDs, visit ```src/experimental.ipynb```.

## Web Application
We have a Streamlit application for XRayPro, available [here](https://xraypro.streamlit.app). If you wish to install the application to run on your machine locally, please visit the [web repository](https://github.com/AI4ChemS/xraypro-web/tree/main) for a guide to install.

## Hardware Requirements
It is highly encouraged for this package to be used on a GPU (for pretraining, finetuning and evaluating data points). This software was developed and tested on NVIDIA RTX4090, so it is highly recommended to use a GPU along this power. Regarding runtime, for finetuning on 4000 data points (CoRE-MOF), it takes around 5-7 minutes to finetune on 100 epochs, whereas for a larger database such as BW20K (20K entries), it takes around 20 minutes to finetune on 30 epochs. 

## Software requirements
### OS Requirements
This package should be working properly on Linux and Windows. In particular, the package has been tested on Ubuntu 22.04.4 LTS.

## Installation
Python 3.11.9 is recommended for this package. Furthermore, when pretraining, finetuning and evaluating the model (especially across many MOFs), a GPU is heavily recommended; please do ``torch.cuda.is_available()`` in your Python environment/notebook to see if your environment is able to correctly access your GPU (if you have one). For complete use of this package, please follow these steps (assuming you have access to conda):

```
git clone https://github.com/AI4ChemS/XRayPro.git
conda create -n xraypro python=3.11.9
conda activate xraypro

cd path/to/xraypro
pip install -r requirements.txt
```

Under the assumption that this is being installed on a fresh environment, the installation time ranges between 2-4 minutes.

## Main

XRayPro is a multimodal model that accepts the powder x-ray diffraction (PXRD) pattern and chemical precursors to make a wide variety of property predictions of metal-organic frameworks (MOFs), while supplementing the most feasible applications per MOF. This is a tool that is motivated by accelerating material discovery. A workflow of our model can be shown below, in which a transformer encodes and embeds the inputted chemical precursor (in the form of the SMILES of the organic linker and metal node), whereas the convolutional neural network (CNN) embeds the PXRD pattern, before performing regression. Furthermore, self-supervised learning (Barlow-Twin) is done on our model against a crystal graph convolutional neural network (CGCNN) to not only improve data efficiency at low data regimes, but also provide more context about the local environment of the MOF. These pretrained weights are loaded into XRayPro and can be finetuned for any task. 

![Methods](https://github.com/user-attachments/assets/72b4d3fc-74bb-4d7f-8ca3-f559f8dfdde0)

### Does this work on any PXRD pattern?

We have evalauted our finetuned model on entries from the Cambridge Structural Database (CSD). As our model was finetuned on CoRE-MOF entries, in which bounded/unbounded solvents are removed from the pores, there was an incentive to assess the robustness on the counterpart entries in which those solvents are still retained when computing the simulated PXRD pattern. This was tested across three different classes: missing hydrogen atoms, bounded and unbounded solvents, showcasing that the model is robust. Furthermore, our model is also robust on experimental PXRD patterns - evaluated on CAU-28, Yb-UiO66 (thank you Ashlee Howarth and team for this data!) and [pyrene-based MOFs](https://pubs.acs.org/doi/full/10.1021/acsami.4c05527).

![CSDAssessment](https://github.com/user-attachments/assets/2598f2f3-04bf-4c2e-b320-5ba25f7e288a)

### Benchmark models

A couple of benchmark models were considered - a descriptor-based ML model (which accepts geometric descriptors and chemistry RACs), CGCNN (which accepts crystal structures) and MOFormer (which accepts MOFids as inputs). It can be seen that our model outperforms MOFormer and CGCNN for geometric properties such as uptake at HP mainly due to the context PXRD patterns provides, alongside competing well against the chemistry-reliant and electronic properties such as band gap. While the descriptor-based ML model generally outperforms XRayPro, the advantage we have is that these descriptors require crystal structures, which are quite challenging to obtain, whereas retrieving the PXRD and chemical precursors are immediately known - ultimately accelerating material discovery. Furthermore, we compete with descriptor-based ML models for geometric properties decently well for this to be sustainable.

The panel on the right shows why we consider multimodality rather than simply using one input. The PXRD and chemical precursor complement well with each other, as the PXRD captures the global structure/environment of the MOF, whereas the chemical precursors describe the metal and organic chemistry. When these two representations are combined, our model is well-rounded and competes with structural models such as CGCNN.

<img width="1186" alt="Figure2_v3" src="https://github.com/user-attachments/assets/cda97bab-6c30-4ed7-8a1c-370f0de939dc" />


## Citation

If you use our work, please cite us using the BibTeX entry below.

```bibtex
@article{khan2024connecting,
  title = {Connecting metal-organic framework synthesis to applications with a self-supervised multimodal model},
  author = {Khan, Sartaaj Takrim and Moosavi, Seyed Mohamad},
  year = {2024},
  journal = {ChemRxiv},
  doi = {10.26434/chemrxiv-2024-mq9b4},
  url = {https://chemrxiv.org/engage/chemrxiv/article-details/671a9d9783f22e42140f2df6},
  note = {Preprint, not peer-reviewed}
}
```

## Privacy when using web application
Our web app tool does **NOT** store any data that is inputted into the entry fields (there is no external database for this).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.

