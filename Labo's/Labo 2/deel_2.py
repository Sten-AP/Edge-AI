import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np

# Stap 1
data_df = pd.read_csv('Labo 2/titanic/train.csv')

# Stap 2
print(data_df.head())
print()
print(data_df.describe())
print()

# stap 3
# data_df.hist()
# plt.title("Histogram van alle data")
# plt.show()
# data_df.boxplot()
# plt.title("Boxplot van alle data")
# plt.show()
# glue = data_df.pivot(index="PassengerId", columns="Sex", values="Survived")
# plt.title("Per passagier, het geslacht en overlevend of niet")
# sns.heatmap(glue)
# plt.show()

# Stap 4
for key in data_df:
    if type(data_df[key][0]) != str and type(data_df[key][0]) != float:
        data_df[key] = data_df[key].fillna(data_df[key].mean())
    else:
        data_df[key] = data_df[key].fillna(data_df[key].mode())

pd.get_dummies(data_df['Sex'], data_df['Embarked'])

scaler = StandardScaler()
scaler.fit([data_df['Age'], data_df['Survived']])

print(scaler.transform([data_df['Age'], data_df['Survived']]))
print()
print(data_df.head)
