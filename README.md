# decodIRT

A IRT application using OpenML database. The objective is to perform the complexity analysis of binary datasets, automatically downloaded from the OpenML platform, and also to compare the performance of several supervised learning algorithms, relating their performances to the instances of the datasets. For this we used the calculation of the IRT item parameters as well as the calculation of the probability of correctness of each classifier for each test instance of a given dataset.

## How to use

decodIRT is a ready-to-use tool, just download and run.

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

<p align="center">
  <img src="https://github.com/LucasFerraroCardoso/IRT_OpenML/raw/master/Fluxograma.png" width="500">
</p>

## Tutorial

Here is a brief tutorial on how to use decodIRT. For details of existing parameters and generated results, see the documentation files.

This tutorial is divided into three parts that correspond to each of the scripts.

### decodIRT_OtML

Although the tool can analyze multiple datasets, in this tutorial only one will be used. The chosen dataset is Credit-g, which can be accessed at https://www.openml.org/d/31.

The first script will be executed using the -data parameter to enter the dataset ID to be downloaded. If more than one dataset, IDs can be passed in list form.

`$ python decodIRT_OtML -data 31`

Information about the chosen dataset and the execution of the classifiers is displayed on screen during program execution. After execution, all generated data will be saved in 6 different files in a directory called "output".

The output directory can be changed if the script is executed with the -output parameter.

### decodIRT_MLtIRT

This script is in charge of calculating IRT parameters using the R language. But don't worry! The script takes care of installing the R packages, you only need Python.

To run this script, just enter the directory where the generated files were saved and the script does the rest.

`$ python decodIRT_MLtIRT -dir dir_name`

In this case the default name (output) was left for the output directory in decodIRT_OtML so it would not be necessary to pass the -dir parameter. The result of this script will be a file containing the TRI item parameters for 30% of the dataset that was used for testing after a 70/30 split.

### decodIRT_analysis

The decodIRT_analysis script, as the name already suggests, aims to perform analyzes using TRI and provide the user with graphs that allow the behavior of different classifiers to be visualized on dataset instances. In addition to showing the percentage of instances that have high values of Discrimination, Difficulty and Guessing.

`$ python decodIRT_analysis -dir dir_name -plotAllHist -plotAllCCC -save`

This command will generate all histograms, curve graphs and save them into the directory where the dataset was downloaded. If the -save parameter is not passed, all requested graphics will be shown on the run screen.

There are more parameters that allow the user to control the value limit of each item parameter, the number of histogram bins and for which dataset and item parameter (Discrimination, Difficulty, Guessing) the graphs will be generated.

## Author's Note

I hope this tool can help you realize interesting insights and assist you in your research!
