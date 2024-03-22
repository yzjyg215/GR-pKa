# GR-pKa: A message-passing neural network with retention mechanism for pKa prediction

## Introduction

This repository provides codes and materials associated with the manuscript [GR-pKa: A message-passing neural network with retention mechanism for pKa prediction].
![image](https://github.com/yzjyg215/GR-pKa/blob/master/graph_abstract.tif)

- **GR-pKa** is a novel pka prediction method which takes advantages of multi-fidelity learning, quantum mechanical (QM) properties and retention mechanism.
The prediction process consisted of three steps: molecular graph construction and featurization, message passing with the retention mechanism, and pKa prediction.

We acknowledge the paper [Liu et al (2023). ABT-MPNN: an atom-bond transformer-based message-passing neural network for molecular property prediction. J Cheminform 2023;15(1):29.](https://doi.org/10.1186/s13321-023-00698-9) and the [Chemprop](https://github.com/chemprop/chemprop) repository ([version 1.2.0](https://github.com/chemprop/chemprop/releases/tag/v1.2.0)) which this code leveraged and built on top of.

## Overview 
- ```GR_pKa/```: the source codes of GR-pKa.
- ```features/```: the molecular QM features of all datasets.
- ```data/```: the pre-training dataset, fine-tuning dataset, E-pKa dataset used in GR-pKa.
- ```model/```: the pre-trained model weights of GR-pKa.
- ```prediction/```: the prediction results on SAMPL6, SAMPL7, and E-pKa datasets by GR-pKa.