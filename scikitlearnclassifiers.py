from sklearn import datasets

iris = datasets.load_iris()

# X is input and Y is output. X is data and Y is label here
x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, .5)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(x_train,y_train)

predictions = my_classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
