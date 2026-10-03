import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("class.csv") #Read data from csv file
fig, ax= plt.subplots() #Initialize the plot


#Prevrashayet eti columns v numpy
features = data[["monthly_income", "app_usage_hours"]].to_numpy()
labels = data["bought_premium"].to_numpy()

x_min_income = features[:, 0].min() - 100
x_max_income = features[:, 0].max() + 100
y_min_hours = features[:, 1].min() - 5
y_max_hours = features[:, 1].max() + 5  #Sozdayet area dlya kvadrata
#Potom mi etot area tam x eto left i right a y eto up and down potom mi etot kvadrat sdelayem bolee chetkim v circle


x_values = np.linspace(x_min_income, x_max_income, 300)
y_values = np.linspace(y_min_hours, y_max_hours, 300)
#mojno skazat sozdayet line graph kotoriy divide na 300 kuskov delayet
#vot eto gde uje on v circle prevrashaytsa

#Sozdayet sam classification model
model = KNeighborsClassifier(n_neighbors=9)
model.fit(features, labels)#uchitsa data

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

ax.scatter(features[:, 0], features[:, 1], c=labels, edgecolors='black') #prosto scatter original data
ax.contourf(xx, yy, borders, alpha=0.3) #sozdat area of classification
plt.tight_layout()
plt.show()

for i in range(5): #predskazivayet teper category
    income = int(input("Enter person's income: "))
    usage_hours = int(input("Enter person's usage hours: "))
    data_predict = np.array([income, usage_hours])
    print(model.predict([data_predict]))



