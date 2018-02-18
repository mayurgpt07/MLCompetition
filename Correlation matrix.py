import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

values = pd.read_csv('F:\Downloads/train_values.csv')
labels = pd.read_csv('F:\Downloads/train_labels.csv')

df = pd.merge(values, labels, on='row_id')

df.drop('row_id', axis=1, inplace=True)

corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


f, ax = plt.subplots(figsize=(11, 11))
cmap = sns.diverging_palette(220, 10, as_cmap=True)


sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.show()
