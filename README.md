![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Open in Streamlit](https://img.shields.io/badge/Streamlit-Open%20in%20Streamlit-brightgreen)](https://<streamlit-link>)

## XRayPro

A recommendation system for MOFs leveraging only PXRDs and precursors. For further details, please refer to our [paper](https://chemrxiv.org/engage/chemrxiv/article-details/671a9d9783f22e42140f2df6). 

## Usage
A demo of the finetuning can be found in ```src/demo.ipynb``` (full finetuning script is available in ```src/run.py```), in which the model is finetuned to predict methane uptake at HP on CoRE-MOF database. For pretraining purposes, please refer to ```src/pretrain.py```. 

## Web Application
We have a Streamlit application for XRayPro, available [here](xraypro.streamlit.app). If you wish to install the application to run on your machine locally, please visit the [web repository](https://github.com/AI4ChemS/xraypro-web/tree/main) for a guide to install.

## Main

XRayPro is a multimodal model that accepts the powder x-ray diffraction (PXRD) pattern and chemical precursors to make a wide variety of property predictions of metal-organic frameworks (MOFs), while supplementing the most feasible applications per MOF. This is a tool that is motivated by accelerating material discovery. A workflow of our model can be shown below, in which a transformer encodes and embeds the inputted chemical precursor (in the form of the SMILES of the organic linker and metal node), whereas the convolutional neural network (CNN) embeds the PXRD pattern, before performing regression. Furthermore, self-supervised learning (Barlow-Twin) is done on our model against a crystal graph convolutional neural network (CGCNN) to not only improve data efficiency at low data regimes, but also provide more context about the local environment of the MOF. These pretrained weights are loaded into XRayPro and can be finetuned for any task. 

![Methods](https://github.com/user-attachments/assets/72b4d3fc-74bb-4d7f-8ca3-f559f8dfdde0)

### Does this work on any PXRD pattern?

We have evalauted our finetuned model on entries from the Cambridge Structural Database (CSD). As our model was finetuned on CoRE-MOF entries, in which bounded/unbounded solvents are removed from the pores, there was an incentive to assess the robustness on the counterpart entries in which those solvents are still retained when computing the simulated PXRD pattern. This was tested across three different classes: missing hydrogen atoms, bounded and unbounded solvents, showcasing that the model is robust. Furthermore, our model is also robust on experimental PXRD patterns - evaluated on CAU-28, Yb-UiO66 (thank you Ashlee Howarth and team for this data!) and [pyrene-based MOFs](https://pubs.acs.org/doi/full/10.1021/acsami.4c05527).

![CSDAssessment](https://github.com/user-attachments/assets/2598f2f3-04bf-4c2e-b320-5ba25f7e288a)

### Benchmark models

A couple of benchmark models were considered - a descriptor-based ML model (which accepts geometric descriptors and chemistry RACs), CGCNN (which accepts crystal structures) and MOFormer (which accepts MOFids as inputs). It can be seen that our model outcompetes MOFormer and CGCNN for geometric properties such as uptake at HP mainly due to the context PXRD patterns provides, alongside competing well against the chemistry-reliant and electronic properties such as band gap. While the descriptor-based ML model generally outperforms XRayPro, the advantage we have is that these descriptors are very difficult to obtain, whereas PXRDs and precursors are very easy to obtain! Furthermore, we compete with descriptor-based ML models for geometric properties decently well for this to be sustainable.

![spider_w_cgcnn_v2](https://github.com/user-attachments/assets/cc690a38-6856-49b9-87e3-111344343148)

## Citation

If you use our work, please cite us using the BibTeX entry below.

```bibtex
@article{XRayPro,
  title={Connecting metal-organic framework synthesis to applications with a self-supervised multimodal model},
  author={Sartaaj Khan and Seyed Mohamad Moosavi},
  journal={journalName},
  year={2024},
  volume={volumeName},
  pages={pagesOfJournal},
  publisher={publisherName}
}
```

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.

