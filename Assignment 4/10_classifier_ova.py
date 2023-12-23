import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from itertools import combinations
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
args=sys.argv
train_df = pd.read_csv('penguins_train.csv')
print(args[1])
test_df=pd.read_csv(args[1])
train_df.head()

X_train, X_test, y_train, y_test = train_test_split(train_df.drop('Species', axis=1), train_df['Species'], test_size=0.3, random_state=42)


def preprocess(X_train):
    # Fill the missing values in 'Clutch Completion' column with the mode
    X_train['Clutch Completion'] = X_train['Clutch Completion'].fillna(X_train['Clutch Completion'].mode()[0])
    #X_train.drop('Sex', axis=1, inplace=True)
    X_train['Sex'] = X_train['Sex'].map({'MALE':1,'FEMALE':0})
    X_train['Sex'].fillna(X_train['Sex'].mode()[0], inplace=True)   
    # Convert the categorical 'Island' column to numerical using LabelEncoder
    le = LabelEncoder()
    X_train['Island'] = le.fit_transform(X_train['Island'])
    # Convert the 'Clutch Completion' column to numerical
    X_train['Clutch Completion'] = X_train['Clutch Completion'].map({'Yes': 1, 'No': 0})
    # Fill the missing values in numerical columns with their mean
    num_cols = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Delta 15 N (o/oo)', 'Delta 13 C (o/oo)']
    X_train[num_cols] = X_train[num_cols].fillna(X_train[num_cols].mean())
    # Scale the numerical columns using StandardScaler
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    return X_train

le = LabelEncoder()
X_train = preprocess(X_train)
y_train = le.fit_transform(y_train)
X_test = preprocess(X_test)
y_test = le.fit_transform(y_test)


classes = np.unique(y_train)
svm_classifiers = {}
for c in classes:
    X_train_class = X_train
    y_train_class = np.where(y_train == c, 1, -1)
    svm = SVC(kernel='linear')
    svm.fit(X_train_class, y_train_class)
    svm_classifiers[c] = svm

# to predict the class of each test data point
y_pred_ova=[]
for i, row in X_test.iterrows():
    scores={}
    for c, svm in svm_classifiers.items():
        scores[c]=svm.predict([row.values])
    y_pred_ova.append(max(scores, key=scores.get))

acc_ova=accuracy_score(y_test,y_pred_ova)
# Print the accuracy of the classifier
print(f"One-vs-All SVM Classifier Accuracy: {acc_ova}")

# Print the confusion matrix, precision, recall, and F1 score metrics
cm = confusion_matrix(y_test, y_pred_ova, labels=classes)
print(f"Confusion Matrix:\n{cm}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_ova, labels=classes)}")

# Plot the confusion matrix
# plot_confusion_matrix(svm, X_test, y_test, labels=classes)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_ova, labels=classes)
plt.show()


# now predict the labels for the test data
test_df = preprocess(test_df)

y_pred_test_ova=[]
for i, row in test_df.iterrows():
    scores={}
    for c, svm in svm_classifiers.items():
        scores[c]=svm.decision_function([row.values])
    y_pred_test_ova.append(max(scores, key=scores.get))

y_pred_test_ova = le.inverse_transform(y_pred_test_ova)
#should output predicted labels in a new file named ova.csv
pd.DataFrame({"predicted":y_pred_test_ova}).to_csv("ova.csv", header=True, index=None)