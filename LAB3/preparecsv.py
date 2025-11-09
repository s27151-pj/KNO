import pandas as pd
from sklearn.utils import shuffle

cols = [
    "Class","Alcohol","Malic_acid","Ash","Alcalinity_of_ash","Magnesium",
    "Total_phenols","Flavanoids","Nonflavanoid_phenols","Proanthocyanins",
    "Color_intensity","Hue","OD280/OD315_of_diluted_wines","Proline"
]

df = pd.read_csv("wine/wine.data", header=None, names=cols)
df = shuffle(df, random_state=42) # [Z1] tasowanie — tylko do podglądu/eksportu
df.to_csv("wine.csv", index=False)
print("saved wine.csv")
