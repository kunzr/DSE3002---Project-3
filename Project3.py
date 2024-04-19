import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import display

tree = pd.read_csv('tree1.csv')
tree = tree.dropna()

X = tree.iloc[:, 0:23]
y = tree.iloc[:, -1]
y_label = y.values

def unique(lists):
    unique_list = pd.Series(lists).drop_duplicates().tolist()
    for x in unique_list:
        print(x)

display(tree)


mms=MinMaxScaler()
X_minmax = mms.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y_label, train_size=0.8, stratify=y_label, random_state=0)

# Initialize kNN classifier
k = 5  # Number of neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Predict labels for test set
y_pred = knn_classifier.predict(X_test)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train) #Fit on the training set (need both data and labels)
y_pred =clf.predict(X_test) #Predict on test set (no labels)
cf_matrix = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cf_matrix, display_labels=['0', '1'])
cmd.plot()
cmd.ax_.set(xlabel='Predicted', ylabel='Actual')
plt.show()

# Calculate F1 score
f1 = f1_score(y_test, y_pred)

print("F1 Score:", f1)


from sklearn.model_selection import GridSearchCV 

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 3, 5, 7, 9]} #set parameters for GridSearch
svc = SVC() #create your SVM object
clf = GridSearchCV(svc, parameters) #create your GridSearch object (which will use your SVM object)
clf.fit(X_train, y_train) #Fit the GridSearch object
print(clf.best_estimator_) 
y_pred = clf.predict(X_test) #Generate a prediction using the model found by GridSearch

cf_matrix = confusion_matrix(y_test,y_pred)
cmd = ConfusionMatrixDisplay(cf_matrix, display_labels=['low quality', 'high quality'])
cmd.plot()
cmd.ax_.set(xlabel='Predicted', ylabel='True')


