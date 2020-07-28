Here you can find all the files and supplementary materials for the paper "Decoding machine learning benchmarks", published in BRACIS20. The files were organized as follows:

### Results BRACIS
- This folder concentrates all the results generated that are used in the paper.  
**datasets.csv**: List of ID's that the datasets used in the paper have in OpenML. This list serves as input for the first script.  
**clf_rating.csv**: File containing the classifier ranting ranking that is shown in Table 1 of the paper.  
**Real_clf_nemenyi.csv:** P-value matrix resulting from the Nemenyi calculation for the real classifiers.  
**IRT_param_freq.txt**: File that shows the percentages of difficult, discriminating and easy-to-guess instances for all datasets.  
**modelosML.txt**: File that lists all hyperparameters used in ML models that were analyzed in the paper.  
**Fluxograma.png**: Flowchart of the decodIRT execution, shown in Figure 1 of the paper.  
**graph_percIRT.png**: Image of the graph that compares the percentages of difficult and discriminating instances of the datasets, shown in Figure 2 of the paper.  
**jm1_score.png**: Image of the comparison chart between the True-Score obtained by the classifiers in the "jm1" dataset, shown in Figure 3 of the paper.  
**heatmap_realclf.png**: Image of the heapmap used to analyze the results of the Nemenyi test, shown in Figure 4 of the paper.  

### Output
- This folder contains all the results generated for each dataset after the execution of each of the three scripts. All results are divided into folders named after each dataset. Each folder contains the following files:

	- Results of the decodIRT_OtML script:  
	     **dataset_name.csv**: The file without suffix indicates that its content is the answer of each of the ML models of the real classifiers and the artificial classifiers.  
	     **dataset_name_acuracia.csv**: The file with the suffix “_acuracia” means that its content is composed of a table containing the average accuracy of each real classifier, during cross-validation.  
	     **dataset_name_final.csv**: The file with the suffix “_final” means that its content consists of a table containing the accuracy of the real classifiers on the separate instances for testing.  
	     **dataset_name_irt.csv**: Just like the file without suffix in the name (dataset_name.csv), this file has an array of responses. However, it does contain a response vector for all real, artificial and MLP classifiers. This matrix is ​​used to generate the IRT item parameters in the second script.  
	     **dataset_name_mlp.csv**: Contains the final accuracy that the first set MLP’s classifiers obtained after the classification.  
	     **dataset_name_test.csv**: The contents of the file are a list of all instances of data that are part of the test set.  

	- Results of the decodIRT_MLtIRT script:  
	     **irt_item_param.csv**: Table containing all item parameters (Difficulty, Discrimination and Guessing) generated for the test set instances.  

	- Results of the decodIRT_analysis script:  
	     **score_disPositivo.csv**: Table containing the True-Score score obtained for each real classifier, considering only the instances with positive discrimination.  
	     **score_total.csv**: Table containing the True-Score score obtained for each real classifier, considering all instances.  
	     **theta_list.csv**: Table that shows the final Theta value obtained by each real classifier.  
	     **dataset_name_score.png**: Image of the comparison chart between the True-Score obtained by the real and artificial classifiers in the dataset.  

### Scripts
- This folder contains all the scripts used to generate the results presented in the paper.

	- decodIRT:  
	    **decodIRT_OtML.py**  
	     **decodIRT_MLtIRT.py**  
	     **decodIRT_analysis.py**  

	- Other Scripts:  
	     **clf_rating_nemenyi.py**: Script created to calculate the rating of the classifiers using the Glicko-2 system and to perform the Friedman and Nemenyi Tests.  
	     *Note: to calculate the ratings it is necessary to have the python script of the Glicko-2 system which can be downloaded through the link http://www.glicko.net/glicko.html
