import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("student_exam.csv") #Read data from csv file
fig, ax= plt.subplots() #Initialize the plot


#Prevrashayet eti columns v numpy
features = data[["study_hours", "sleep_hours"]].to_numpy()
labels = data["passed"].to_numpy()

x_min = features[:, 0].min() - 0.5
x_max= features[:, 0].max() + 0.5
y_min= features[:, 1].min() - 0.5
y_max= features[:, 1].max() + 0.5  #Sozdayet area dlya kvadrata
#Potom i etot area tam x eto left i right a y eto up and down potom mi etot kvadrat sdelayem bolee chetkim v circle


x_values = np.linspace(x_min, x_max, 300)
y_values = np.linspace(y_min, y_max, 300)
#mojno skazat sozdayet line graph kotoriy divide na 300 kuskov delayet
#vot eto gde uje on v circle prevrashaytsa

#Sozdayet sam classification model

training_accuracy = {}

for k in range(1,31,2):
    training_accuracy[k] = cross_val_score(KNeighborsClassifier(n_neighbors=k), features, labels, cv=5).mean()
print(training_accuracy)

max_value = max(training_accuracy.values())
num = None
for key, value in training_accuracy.items():
    if value == max_value:
        print(key)
        num = key

model = KNeighborsClassifier(n_neighbors=num)

model.fit(features, labels)
#vot eto sozdayet coordinates dlya kajdogo value in each array
xx, yy = np.meshgrid(x_values, y_values)

grid_points = np.c_[xx.ravel(), yy.ravel()]
#Eto creates array s coordinates for each xx and yy index
#tipo sozdayet parochku gde xx[0], yy[0]

predictions = model.predict(grid_points)#ispolzaya eti grid points
#mi mojem nayti borders by looking where 1 and 0 are near each other

borders = predictions.reshape(xx.shape)
#naxodit borders smotrya na predictions i reshape delaet chtob
#array shape bil kak coordinate
ax.set_xlabel("study_hours")
ax.set_ylabel("sleep_hours")
ax.scatter(features[:, 0], features[:, 1], c=labels, edgecolors='black') #prosto scatter original data
ax.contourf(xx, yy, borders, alpha=0.3) #sozdat area of classification
plt.tight_layout()
plt.show()

for i in range(5): #predskazivayet teper category
    income = int(input("Enter person's study hours: "))
    usage_hours = int(input("Enter person's sleep hours: "))
    data_predict = np.array([income, usage_hours])
    print(model.predict([data_predict]))



