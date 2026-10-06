from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("student_exam.csv")
fig, ax = plt.subplots()

feature = data["study_hours"].values
label = data["exam_score"].values

x = feature.reshape(-1,1) #scikit-learn needs it reshaped because linear regression
#may have more than 1 feature
model = LinearRegression()
rng = np.random.default_rng()
random_states = rng.integers(1,100, size=20)
error_count = 0
scores = []
for random_state in random_states:
    x_train, x_test, y_train, y_test = train_test_split(feature, label,
                                                        test_size=0.2,
                                                        random_state=random_state)
    x_reshaped = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    model = clone(model)
    model.fit(x_reshaped, y_train)
    error_count += 1
    mae_y_pred = model.predict(x_test)
    print(f"Test number {error_count}:")
    print(f"Score: {model.score(x_test, y_test)}")
    scores.append(model.score(x_test, y_test))
    print(f"MAE: {mean_absolute_error(y_test, mae_y_pred)}")
    print(f"MSE: {mean_squared_error(y_test, mae_y_pred)}")
    print(f"RMSE: {root_mean_squared_error(y_test, mae_y_pred)}")

print("Scores with train_test_split")
print(scores)
print(np.mean(scores))
print("Cross Val Score")

model = clone(model)
model.fit(x, label)

cv_scores = cross_val_score(model, x, label, cv=5)
print(cv_scores)
print(cv_scores.mean())

print(f"Gradient: {model.coef_}") #Gradient of regression
print(f"Y Intercept: {model.intercept_}") #Y intercept of regression

ax.scatter(feature, label, color="blue")
ax.set_xlabel("Study Hours")
ax.set_ylabel("Exam Score")

line = model.predict(x) #when giving same data before it can give us line to use a trick
ax.plot(feature, line, color="orange", linewidth=4)

plt.tight_layout()
plt.show()


for i in range(5):
    hours = input("Enter hours studied: ")
    print(f"You will probably get: {model.predict(hours)}")
