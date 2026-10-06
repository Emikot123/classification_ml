import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('student_exam.csv')
fig, ax = plt.subplots()

feature = data["study_hours"].values
label = data["passed"].values
print(feature[0:5])
train = feature.reshape(-1,1)

model = LogisticRegression()
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

