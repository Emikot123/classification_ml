from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("student_exam.csv")
fig, ax = plt.subplots()
model = LinearRegression()

feature = data["study_hours"].values
label = data["exam_score"].values
x = feature.reshape(-1,1)
model.fit(x, label)

print(f"Gradient: {model.coef_}")
print(f"Y Intercept: {model.intercept_}")

ax.scatter(feature, label, color="blue")
ax.set_xlabel("Study Hours")
ax.set_ylabel("Exam Score")

line = model.predict(x)
ax.plot(feature, line, color="orange", linewidth=4)

plt.tight_layout()
plt.show()