import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from source.data_preprocess import DataPreprocessing

class ModelBuilder2(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder2, self).__init__(*args, **kwargs)

    def ann(self, X_train,X_test,y_train,y_test):

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
        
        clf.fit(X_train)
        
        ann_predicted = clf.predict(X_test)

        error = 0
        for i in range(len(y_test)):
            error += np.sum(ann_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        self.accuracy = accuracy_score(y_test,ann_predicted)

        return clf