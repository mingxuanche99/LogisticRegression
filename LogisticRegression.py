import numpy as np
import matplotlib.pyplot as plt
def sigmod(x):
    return 1/(1+np.exp(-x))
class LogisticRegression:
    def __init__(self,  learn_rate=0.0001, iter=1000):
        self.learn_rate=learn_rate
        self.iter=iter
        self.W=0
        self.b=0

    def fit(self,X_train,Y_train):
        self.W=np.zeros((X_train.shape[1],1))
        X_train = X_train.T
        Y_train = Y_train.reshape(1,Y_train.shape[0])
        loss_record=[]
        for i in range(self.iter):
            # forward proagate
            Y=sigmod(np.dot(self.W.T,X_train)+self.b)
            #calculate cost
            cost=-(np.sum(Y_train*np.log(Y)+(1-Y_train)*np.log(1-Y)))/X_train.shape[1]
            loss_record.append(cost)
            #calculate gradient
            dz=Y-Y_train
            gradient_w=(np.dot(X_train,dz.T))/X_train.shape[1]
            gradient_b=(np.sum(dz))/X_train.shape[1]
            #gradient decent
            self.W=self.W-self.learn_rate*gradient_w
            self.b=self.b-self.learn_rate*gradient_b
        #plot cost effect curve
        print(loss_record)
        plt.figure()
        plt.plot(loss_record, 'b', label='loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
    def predict(self,X):
        pass

    def score(self,X_test,Y_test):
        pass





