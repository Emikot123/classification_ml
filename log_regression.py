import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv('student_exam.csv')
fig, ax = plt.subplots()

feature = data["study_hours"].values
label = data["passed"].values
train = feature.reshape(-1,1)

model = LogisticRegression()

rng = np.random.default_rng()
random_states = rng.choice(100,size=5, replace=False)
error_count = 0
for random_state in random_states:
    x_train, x_test, y_train, y_test = train_test_split(
        feature, label, test_size=0.2, random_state=random_state, stratify=label
    )
    model = clone(model)
    x_reshaped = x_train.reshape(-1,1)
    x_test_reshaped = x_test.reshape(-1,1)
    error_count += 1
    model.fit(x_reshaped, y_train)
    y_pred = model.predict(x_test_reshaped)
    y_prob_pred = model.predict_proba(x_test_reshaped)[:, 1]
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Test Count {error_count}: ")
    print(f"Score: {model.score(x_test_reshaped, y_test)}")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")#same as mode.score just did it for learning
    print(f"Matrix Confusion: {conf_matrix}")
    print(f"Classification Report: {classification_report(y_test, y_pred)}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob_pred)}")

model = clone(model)
model.fit(train, label)
print(f"Cross Val Score: {cross_val_score(model, train, label)}: ")
line_values = np.linspace(feature.min(), feature.max(), 200).reshape(-1,1)
line = model.predict_proba(line_values)[:, 1]

ax.plot(line_values, line, color='red')

ax.scatter(feature, label, color="blue")

plt.tight_layout()
plt.show()

for _ in range(5):
    hours = int(input("enter how much hours studied: "))
    hours = np.array([hours])
    pred = hours.reshape(-1, 1)
    print(f"Probability of you passing {model.predict_proba(pred)}")

