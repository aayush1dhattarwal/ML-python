# Dependencies
from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB

# Train the Data [height, weight, shoe size]
X = [[180, 75, 39], [170, 58, 31], [165, 53, 29], [186, 74, 41], [154, 61, 26], [191, 85, 40], [157, 60, 32], [174, 68, 31], [179, 71, 34], [158, 64, 30]]
# Lables for data X
Y = ['male', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'female']

# Classifies
clf = tree.DecisionTreeClassifier()
clf1 = svm.SVC()
clf2 = neighbors.KNeighborsClassifier()
clf3 = GaussianNB()

# Train Model
clf = clf.fit(X, Y)
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)

# Data to be predicted
X1 = [[181, 76, 40], [171, 59, 28], [164, 51, 29], [190, 88, 38], [180, 78, 41], [174, 101, 41], [150, 52, 29], [161, 61, 34], [187, 68, 38], [157, 80, 34]]
Y1 = ['male', 'male', 'female', 'male', 'male', 'female', 'male', 'female', 'male','female']

# Prediction
prediction = clf.predict(X1)
prediction1 = clf1.predict(X1)
prediction2 = clf2.predict(X1)
prediction3 = clf3.predict(X1)

# Results
r1 = accuracy_score(Y1,prediction1)
r2 = accuracy_score(Y1,prediction2)
r3 = accuracy_score(Y1,prediction3)

# Print Best Results
if r1 > r2 and r1 > r3:
	print('SVM : ',r1)
elif r2 > r1 and r2 > r3:
	print('KNeighbor : ', r2)
else:
	print('GaussianNB : ', r3)