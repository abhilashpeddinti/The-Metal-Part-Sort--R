------------------------------------------------------------------------------------------------------
*****************************Welcome to the Project : The Metal Part SORT-R***************************
------------------------------------------------------------------------------------------------------

Developed by:
Abhilash Peddinti


Instructions on how to run the project

1. Copy the 4 python files trainMLP.py,executeMLP.py,trainDT.py,executeDT.py to your local python environment.
2. Copy the train_data.csv and test_data.csv files into the same python environment.


Multi-Layer Perceptron(MLP)
---------------------------

You need matplotlib,numpy libraries to run this part.

First you need to train the network,for that you need to execute trainMLP.py

command to execute : python3 trainMLP.py train_data.csv <epochs>
you can change the <epochs> as needed
you can replace train_data.csv with your custom file name.
Now, the network will be trained and a plot of SSE vs Epochs will be displayed.

For the other part, you need to execute executeMLP.py to run the trained model.

command to execute : python3 executeMLP.py weights_<epochs>.csv test_data.csv
you can change the <epochs> as needed.

This will display the confusion matrix,profit and the prediction accuracy rate for the trained model.
Sample Output is attached below.

For Epochs = 100 and neuron count = 5
****************************************************************************
----------------------------Accuracy Statistics-----------------------------
****************************************************************************

Accuracy for the testing data is calculated as stated below :
Prediction Accuracy = no. of correct predictions/total predictions(correct+incorrect)
Prediction Accuracy Obtained for test_data.csv : 0.55

****************************************************************************

************************************************************************************
******************************** Confusion Matrix **********************************
************************************************************************************
------------------------------------------------------------------------------------
| Class   |	 Class 1   	|	 Class 2   	|	 Class 3   	|	 Class 4   	|	total  |
------------------------------------------------------------------------------------
| Class 1 |	 0   		|	 0   		|	 0   		|	 0   		|	 0      |
| Class 2 |	 1   		|	 6   		|	 0   		|	 3   		|	 10      |
| Class 3 |	 4   		|	 0   		|	 5   		|	 1   		|	 10      |
| Class 4 |	 0   		|	 0   		|	 0   		|	 0   		|	 0      |
------------------------------------------------------------------------------------
| Total   |	 5   		|	 6   		|	 5   		|	 4   		|	 20      |
------------------------------------------------------------------------------------


***********************************************************************
******************************** Profit *******************************
***********************************************************************

Overall Profit(in cents):  52

***********************************************************************


Second Part : Decision Tree
---------------------------
You need matplotlib,numpy,sys,operator libraries to run this part.

First you need to train the network,for that you need to execute trainDT.py

command to execute : python3 trainDT.py train_data.csv 
you can replace train_data.csv with your custom file name.
Now, the network will be trained.

The program will generate two files with resulting Decision trees before and after Pruning
DecisionTree.csv contains resulting Decision Tree before pruning.
PrunedDecisionTree.csv contains resulting Pruned Decision Tree.

This will also print two plots of decision boundaries before pruning and afer pruning.


SAmple Output.
THE VALUES GENERATED FOR THE DECISION TREE ARE: 

NUMBER OF NODES GENERATED 19
NUMBER OF LEAF NODES GENERATED : 10
MAXIMUM DEPTH OF THE DECISION TREE  4
MINIMUM DEPTH OF THE DECISON TREE  2
AVERAGE DEPTH OF THE DECISION TREE  3.5


 THE VALUES GENERATED FOR THE PRUNED DECISION TREE ARE: 
 
NUMBER OF NODES GENERATED: 7
NUMBER OF LEAF NODES GENERATED : 4
MAXIMUM DEPTH OF THE DECISION TREE:  2
MINIMUM DEPTH OF THE DECISON TREE:  2
AVERAGE DEPTH OF THE DECISION TREE:  2.0


For the other part, you need to execute executeDT.py to run the trained model.

command to execute without pruning : python3 executeDT.py DecisionTree.csv test_data.csv

you can also use PruneDecisionTree.csv instead of DecisionTree.csv to compare the accuracy and profit,which will be same.


This prints the following and also plots a decision boundary graph

-------------------------------------------CONFUSION MATRIX ----------------------------------------

MATRIX  *	 Class 1  	|	 Class 2  	|	 Class 3  	|	 Class 4  	*	Total
Class 1  *	 5     		|	 0     		|	 0     		|	 1     		*	 6    
Class 2  *	 0     		|	 6     		|	 0     		|	 0     		*	 6    
Class 3  *	 0     		|	 0     		|	 5     		|	 0     		*	 5    
Class 4  *	 0     		|	 0     		|	 0     		|	 3     		*	 3    
--------------------------------------------------------------------------------------------------------------

Total    -->	 5     		+	 6     		+	 5     		+	 4     		=	 20   
--------------------------------------------------------------------------------------------------------------


--------------------------------- RECOGNITION RATE --------------------------------------------
TOTAL ACCURACY OF THE CLASSIFICATION :  0.95
 MEAN OF THE EACH CLASS ACCURACY :  0.9375
-------------------------------------------------------------------------------------------------


 -------------------------------------- PROFIT ---------------------------------------
TOTAL PROFIT: 199
----------------------------------------------------------------------------------------


