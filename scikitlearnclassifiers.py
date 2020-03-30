from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

iris = datasets.load_iris()

# X is input and Y is output. X is data and Y is label here
x = iris.data
y = iris.target


x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = .5)


my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(x_train,y_train)

predictions = my_classifier.predict(x_test)


print(accuracy_score(y_test, predictions))
