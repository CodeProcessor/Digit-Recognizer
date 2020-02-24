'''
Created on 2/24/20

@author: dulanj
'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class DigitClassifier(object):

    def __init__(self):
        self.test_data_set = "data/test.csv"
        self.train_data_set = "data/train.csv"
        self.train_data = None
        self.logreg = None

    def train(self):
        self.train_data = pd.read_csv(self.train_data_set)
        # print(self.train_data.columns)
        X = self.train_data.loc[:, self.train_data.columns != "label"]
        Y = self.train_data.loc[:, self.train_data.columns == "label"]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        self.logreg = LogisticRegression(random_state=0)

        self.logreg.fit(X_train, y_train)

        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
            self.logreg.score(X_test, np.ravel(y_test, order='C'))))

    def test(self):
        test_data = pd.read_csv(self.test_data_set)
        y_pred = self.logreg.predict(test_data)
        with open('submission_logistic_reg_1.csv', 'w') as fp:
            fp.write("ImageId,Label\n")
            for i,p in enumerate(y_pred):
                fp.write("{},{}\n".format(i+1, p))


if __name__ == "__main__":
    dc = DigitClassifier()
    dc.train()
    dc.test()
