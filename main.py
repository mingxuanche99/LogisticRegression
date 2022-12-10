from LogisticRegression import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
cancer=datasets.load_breast_cancer()
X=cancer.data
X=X[:,:2]
y=cancer.target
X_train,X_test,y_train,y_test=train_test_split(X,y)
LR=LogisticRegression()
LR.fit(X_train,y_train)
LR.score(X_test,y_test)