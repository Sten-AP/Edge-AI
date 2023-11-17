import pandas as pd
import matplotlib.pyplot as plt

# Stap 1
data_df = pd.read_csv('Labo 2/austpop.csv')

# Stap 2
for key in data_df:
    if key != 'rownames':
        modes = []
        for mode in data_df[key].mode(key):
            modes.append(mode)
        print(
            f"{key}\tgemiddelde: {round(data_df[key].mean(), 2)}\tmediaan: {data_df[key].median()}\t    modus: {modes}")

print("------------------------------------")
for key in data_df:
    if key != 'rownames':
        print(
            f"{key}\tstandaardafwijking: {data_df[key].std()}\t interkwartielafstand: {data_df[key].quantile()}")

# Stap 3
plt.subplot(121)
data_df['NSW'].hist()
plt.title("Histogram van NSW")
plt.subplot(122)
data_df.boxplot('NSW')
plt.title("Boxplot van NSW")
plt.show()

# De gegevens zijn normaal maar er zijn weinig datapunten.

# Stap 4
data_df.hist()
plt.title("Histogram van alle data")
plt.show()
data_df.boxplot()
plt.title("Boxplot van alle data")
plt.show()

# In Aust is de populatie het grootst.
# In ACT is de groei van de populatie het grootst.
# Voor de rest is de populatiegroei redelijk lineair.
