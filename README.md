# decodIRT

A IRT application using OpenML database. The objective is to perform the complexity analysis of binary datasets, automatically downloaded from the OpenML platform, and also to compare the performance of several supervised learning algorithms, relating their performances to the instances of the datasets. For this we used the calculation of the IRT item parameters as well as the calculation of the probability of correctness of each classifier for each test instance of a given dataset.

## How to use

decodIRT is a ready-to-use tool, just download and run :)

Some dependencies need to be installed for the tool to work. They are listed below.

### Dependencies

- OpenML API
- Scikit-Learning
- Numpy
- Pandas
- Matplotlib
- Package rpy2
- Catsim

Once all the necessary packages are installed, we can move on to running the decodIRT tool.

## Running decodIRT

The decodIRT tool is made up of three scripts written in Python, made for sequential use. The image below shows how to use the scripts in sequence.

<img src="https://github.com/LucasFerraroCardoso/IRT_OpenML/raw/master/Fluxograma.png" width="500">

## Tutorial

Here is a brief tutorial on how to use decodIRT. For details of existing parameters and generated results, see the documentation files.

This tutorial is divided into three parts that correspond to each of the scripts.

### decodIRT_OtML

Although the tool can analyze multiple datasets, in this tutorial only one will be used. The chosen dataset is Credit-g, which can be accessed at https://www.openml.org/d/31.
