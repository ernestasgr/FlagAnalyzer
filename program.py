from matplotlib import pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from graphviz import Source
import re
import seaborn as sb
import time

data = pd.read_csv('flags.csv')

X = data.drop(columns=['religion', 'name'])
y = data['religion']

# Encode the categorical attributes as numerical codes
mappings = {}
X_encoded = pd.DataFrame()
for col in X.columns:
    le = LabelEncoder()
    X_temp = X[col]
    X_encoded[col] = le.fit_transform(X_temp)
    mapping = {idx: label for idx, label in enumerate(le.classes_)}
    mappings[col] = mapping

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3)

start_time = time.time()
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
end_time = time.time()

time_taken = end_time - start_time

print(f"Time taken to fit the decision tree: {time_taken:.3f} seconds")

dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=y.unique(), filled=True, rounded=True, special_characters=True, node_ids=True)

for col in X.columns:
    for code, label in mappings[col].items():
        dot_data = re.sub(rf'{col} &le; {code}\..', rf'{col} &le; {label}', dot_data)

graph = Source(dot_data)
graph.render(filename='decision_tree', format='png', cleanup=True, view=True, directory='.', quiet=False)

y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

cm = metrics.confusion_matrix(y_test, y_pred, labels=y.unique())

# Create a heatmap using the confusion matrix
sb.heatmap(cm, annot=True, cmap="Blues", xticklabels=y.unique(), yticklabels=y.unique())

# Set the title and axis labels
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

rf = RandomForestClassifier(n_estimators=8, max_depth=5)
rf.fit(X_train, y_train)
y_predict = rf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_predict))

for i in range(len(rf.estimators_)):
    tree_data = export_graphviz(rf.estimators_[i], out_file=None, feature_names=X.columns, class_names=y.unique(), filled=True, rounded=True, special_characters=True, node_ids=True)
    tree_src = Source(tree_data)
    tree_src.render('tree_{}'.format(i), format='png', cleanup=True)

    rf.estimators_[i].fit(X_train, y_train)
    y_pred = rf.estimators_[i].predict(X_test)
    print(f"Tree {i} Accuracy:", metrics.accuracy_score(y_test, y_pred))


# Show the plot
plt.show()
