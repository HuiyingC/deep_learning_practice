
from random import randrange
from random import randint
import numpy as np
from decision_tree import DecisionTree

# fold size (% of dataset size) e.g. 3 means 30%
FOLD_SIZE = 10
# number of trees
N_TREES = 20
# max tree depth
MAX_DEPTH = 30
# min size of tree node
MIN_NODE = 1


class RandomForest:
    def __init__(self,n_trees,fold_size):
        self.n_trees = n_trees
        self.fold_size = fold_size
        self.trees = list()



    """
        This function splits the given dataset into n-folds with replacement. The number of folds is equal to the number of the trees that will be trained.
        Each tree will have one fold as input. The size of the folds is a percentage (p) of the size of the original dataset. 

        Parameters:
        dataset: np array of the given dataset
        n_folds (int): number of folds in which the dataset should be split. Must be equal to the number of trees the user wants to train
        p (int): suggests the percentage of the dataset's size the size of a single fold should be.

        Returns list of np arrays: list with the k-folds 

    """
    def cross_validation_split(self,dataset, n_folds, p):
        dataset_split = list()
        fold_size = int(len(dataset)*p/10)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset))
                fold.append(dataset[index])
            set = np.array(fold)
            dataset_split.append(set)
        return dataset_split


    """
        This function randomizes the selection of the features each tree will be trained on.

        Parameters:
            splits list of np arrays: list of folds
            
        Returns list of np arrays: list with the k-folds with some features randomly removed

    """
    def randomize_features(self,splits):
        dataset_split = list()
        l = len(splits[0][0])
        n_features = int((l-1)*5/10)
        for split in splits:
            for i in range(n_features):
                rng = list(range(len(split[0]) - 1))
                selected = rng.pop(randint(0,len(rng)-1))
                split = np.delete(split, selected, 1)
            set = np.array(split)
            dataset_split.append(set)
        return dataset_split


    def print_trees(self):
        i = 1
        for t in self.trees:
            print("Tree#",i)
            temp = t.final_tree
            t.print_dt(temp)
            print("\n")
            i = i+1


    def train(self,X):
        train_x = self.cross_validation_split(X,self.n_trees,self.fold_size)
        train_x = self.randomize_features(train_x)
        for fold in train_x:
            dt = DecisionTree(MAX_DEPTH, MIN_NODE)
            dt.train(fold)
            self.trees.append(dt)



    def predict(self,X):
        predicts = list()
        final_predicts = list()
        for tree in self.trees:
            predicts.append(tree.predict(X))
        # iterate through each tree's class prediction and find the most frequent for each instance
        for i in range(len(predicts[0])):
            values = list()
            for j in range(len(predicts)):
                values.append(predicts[j][i])
            final_predicts.append(max(set(values), key=values.count))
        return final_predicts,predicts



if __name__ == "__main__":


    # Training data
    train_data = np.loadtxt("example_data/data.txt", delimiter=",")
    train_y = np.loadtxt("example_data/targets.txt")

    mock_train = np.loadtxt("example_data/mock_data.csv", delimiter=",")
    mock_y = mock_train[ : , -1]

    # Build and train model
    rf = RandomForest(N_TREES,FOLD_SIZE)
    rf.train(mock_train)

    # Evaluate model on training data
    y_pred,y_pred_ind = rf.predict(mock_train)
    print(f"Accuracy of random forest: {sum(y_pred == mock_y) / mock_y.shape[0]}")
    print("\nAccuracy for each individual tree:")
    c = 1
    for i in y_pred_ind:
        print("\nTree",c)
        print(f"Accuracy: {sum(i == mock_y) / mock_y.shape[0]}")
        c = c+1
