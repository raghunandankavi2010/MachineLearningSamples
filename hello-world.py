from sklearn import tree
features = [[140,1],[130,1],[150,0],[170,0]]
labels = ["0","0","1","1"]
##Dececion tree like if else
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
print(clf.predict([[150,0]]))