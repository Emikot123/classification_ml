from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("student_exam.csv")
fig, ax = plt.subplots()
model = LinearRegression()

feature = data["study_hours"].values
label = data["exam_score"].values

x = feature.reshape(-1,1) #scikit-learn needs it reshaped because linear regression
#may have more than 1 feature
model.fit(x, label)

print(f"Gradient: {model.coef_}") #Gradient of regression
print(f"Y Intercept: {model.intercept_}") #Y intercept of regression

ax.scatter(feature, label, color="blue")
ax.set_xlabel("Study Hours")
ax.set_ylabel("Exam Score")

line = model.predict(x) #when giving same data before it can give us line to use a trick
ax.plot(feature, line, color="orange", linewidth=4)

plt.tight_layout()
plt.show()

print('Error Handling: ')

for i in range(5):
    hours = input("Enter hours studied: ")
    print(f"You will probably get: {model.predict(hours)}")
