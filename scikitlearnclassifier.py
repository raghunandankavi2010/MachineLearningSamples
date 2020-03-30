from scipy.spatial import distance
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def euc(a, b):
    return distance.euclidean(a, b)


class ScrappyKNN:

    def __init__(self):
        self.X_train = None
        self.y_train = None

    def fit(self, X_train_new, y_train_new):
        self.X_train = X_train_new
        self.y_train = y_train_new

    def predict(self, X_test_new):
        predictions_array = []
        for row in X_test_new:
            label = self.closest(row)
            predictions_array.append(label)

        return predictions_array

    def closest(self, row):
        # Distance from test point to first training point
        best_dist = euc(row, self.X_train[0])  # Shortest distance found so far
        best_index = 0  # index of closest training point
        for i in range(1, len(self.X_train)):  # Iterate over all other training points
            dist = euc(row, self.X_train[i])
            if dist < best_dist:  # Found closer, update
                best_dist = dist
                best_index = i
        return self.y_train[best_index]  # closest example


iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

# alternate kneightborsclassifier
# from sklearn.neighbors import KNeighborsClassifier

my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

print(accuracy_score(y_test, predictions))
