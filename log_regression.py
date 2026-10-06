import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('student_exam.csv')
fig, ax = plt.subplots()

feature = data["study_hours"].values
label = data["passed"].values
print(feature[0:5])
train = feature.reshape(-1,1)

model = LogisticRegression()

rng = np.random.default_rng()
random_states = rng.integers(1,100,size=5)
error_count = 0
for random_state in random_states:
    x_train, x_test, y_train, y_test = train_test_split(
        feature, label, test_size=0.2, random_state=random_state,
    )
    model = clone(model)
    x_reshaped = x_train.reshape(-1,1)
    x_test_reshaped = x_test.reshape(-1,1)
    error_count += 1
    model.fit(x_reshaped, y_train)
    print(f"Test Count {error_count}: ")
    print(x_test_reshaped[0:5])
    print(y_test[0:5])
    print(f"Score: {model.score(x_test_reshaped, y_test)}")


model.fit(train, label)

line_values = np.linspace(feature.min(), feature.max(), 200).reshape(-1,1)
line = model.predict_proba(line_values)[:, 1]

ax.plot(line_values, line, color='red')

ax.scatter(feature, label, color="blue")

plt.tight_layout()
plt.show()

for i in range(5):
    hours = input("Enter how much hours studied: ")
    print(f"Probability of you passing {model.predict(hours)}")

