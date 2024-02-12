import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

glass_data = pd.read_csv('glass.csv')
features = glass_data.drop(['Type'], axis=1)
labels = glass_data["Type"]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = LinearSVC(random_state=42)
model.fit(X_train, y_train)

predicted_labels = model.predict(X_test)
accuracy = model.score(X_test, y_test)
classification_report_result = classification_report(y_test, predicted_labels)

print("Accuracy Score: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n", classification_report_result)

