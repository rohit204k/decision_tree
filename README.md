# Decision Tree - From Scratch
## Simple implementation of Decision Tree Algorithm

Decision Tree are very easy to understand and explainable supervised learning algorithm. They have a high variance and not robust.
This is a very simple implementation of Decision Tree algorithm designed for learning purpose.

Note - All the attributes in the dataset are categorical, this algorithm does not handle numerical attributes. To handle numerical attributes please refer Random Forest algorithms, the decision tree implemention used for random forest handles numerical attributes.

## Features
- The dataset file to run the code is present in the repo. Alternatively, any other .csv file can also be provided.
- To calculate the information gain, I have used ID3 as well as Gini Criterion. Please refer the individual files for the respective implemention.
- To test the model, I trained the model using train set and tested it on both train and test set, repeated the experiment 20 times and averaged the results.
- I have plotted accuracy of model over training and testing data.
- With different data in the train set, the tree tends to be different. This can be studied from the graph.
- The stopping criterion used here is - 1. When no more attributes are left to test for gain.
                                        2. When no more training data is left to split.
                                        3. More stopping criterion can be added as per to requirement.

## Steps to run the code

1. The .ipynb file can be run through jupyter notebook
2. The .py file needs to be run through the following commands.
    * Ensure that all requirements have been installed using
        ```sh
            pip install -r requirements.txt
        ```
    * Run the file using
        ```sh
            python decision_tree.py
        ```
