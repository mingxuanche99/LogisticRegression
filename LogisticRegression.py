import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/(1+np.exp(-x))
class LogisticRegression:
    def __init__(self,  learn_rate=0.001, iter=10000):
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
            Y=sigmoid(np.dot(self.W.T,X_train)+self.b)
            #calculate cost
            cost=-(np.sum(Y_train*np.log(Y)+(1-Y_train)*np.log(1-Y)))/X_train.shape[1]
            loss_record.append(cost)
            #calculate gradient
            gra=Y-Y_train
            gradient_w=(np.dot(X_train,gra.T))/X_train.shape[1]
            gradient_b=(np.sum(gra))/X_train.shape[1]
            #gradient decent
            self.W=self.W-self.learn_rate*gradient_w
            self.b=self.b-self.learn_rate*gradient_b
        #plot cost effect curve
        plt.figure()
        plt.plot(loss_record, 'b', label='loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
    def predict(self,X):
        X_train=X.T
        Y = sigmoid(np.dot(self.W.T, X_train) + self.b)
        return Y


    def score(self,X_test,Y_test):
        '''
        X: (num_sample, dim)
        W: (dim, 1)
        Y: (num_sample, )
        '''

        n_sample, n_dim = X_test.shape

        Y_pred = sigmoid(np.dot(self.W.T, X_test.T) + self.b).squeeze(0)
        Y_pred = np.around(Y_pred).astype(np.uint8)

        acc = (Y_pred == Y_test).sum() / n_sample * 100

        if n_dim == 2:
            axis_x = X_test[:,0]
            axis_y = X_test[:,1]

            axis_x_gt_0 = axis_x[Y_test == 0]
            axis_x_gt_1 = axis_x[Y_test == 1]
            axis_y_gt_0 = axis_y[Y_test == 0]
            axis_y_gt_1 = axis_y[Y_test == 1]

            axis_x_pred_0 = axis_x[Y_pred == 0]
            axis_x_pred_1 = axis_x[Y_pred == 1]
            axis_y_pred_0 = axis_y[Y_pred == 0]
            axis_y_pred_1 = axis_y[Y_pred == 1]

            ax1 = plt.subplot(1,2,1)
            plt.scatter(axis_x_gt_0, axis_y_gt_0, c='r')
            plt.scatter(axis_x_gt_1, axis_y_gt_1, c='b')
            ax2 = plt.subplot(1,2,2)
            plt.scatter(axis_x_pred_0, axis_y_pred_0, c='r')
            plt.scatter(axis_x_pred_1, axis_y_pred_1, c='b')
            x_points = np.linspace(axis_x.min()-1, axis_x.max()+1, 100)
            y_points = (self.W[0][0] * x_points + self.b) / (-self.W[1][0])
            plt.plot(x_points, y_points, c='g')

            plt.show()

        print("Classification accuracy on test set: %-10.3f" % acc)
        return acc





