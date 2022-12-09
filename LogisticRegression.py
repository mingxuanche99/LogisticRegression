import numpy as np
def sigmod(x):
    return 1/(1+np.exp(-x))
class LogisticRegression:
    def __init__(self, train_x , train_y, test_x, test_y, learn_rate=0.01, iter=10000):
        self.train_x=np.array(train_x)
        self.train_y=np.array(train_y)
        self.test_x=np.array(test_x)
        self.test_y=np.array(test_y)
        self.learn_rate=learn_rate
        self.iter=iter
        self.W=np.zeros((self.train_x.shape[1],1))
        self.b=0
    def train(self):

    def test(self):
