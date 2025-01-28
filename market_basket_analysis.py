# %%
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import time
df = pd.read_csv('OnlineRetailShopGermany.csv')
df.head()

# %%
print (df.isnull().values.any())
missing_value = ["NaN", "NONE", "None", "nan", "none", "n/a", "na", " "]
df = pd.read_csv('OnlineRetailShopGermany.csv', na_values = missing_value)
print (df.isnull().sum())
df['Description'] = df['Description'].str.strip()
# ranking the top 10 best-selling items
df.Description.value_counts(normalize=True)[:10]
df.drop(df[df['Description'] == 'POSTAGE'].index, inplace = True)
df.shape
df.Description.value_counts(normalize=True)[:30].plot(kind="bar", figsize=(10,5), title="Percentage of Sales by Item").set(xlabel="Item", ylabel="Percentage")

# %%
# create a bar chart, rank by value
df.Description.value_counts()[:30].plot(kind="bar", figsize=(10,5), title="Total Number of Sales by Item").set(xlabel="Item", ylabel="Total Number")
df2 = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))
def convertToZeroOne(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

df3 = df2.applymap(convertToZeroOne)
start_time = time.time()
frequent_itemsets = apriori(df3, min_support=0.04, use_colnames=True)
end_time = time.time()
frequent_itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
#Filtering rules based on condition
rules[(rules['lift'] >= 0.5) & (rules['confidence'] >= 0.3)]

# %%
# Import seaborn under its standard alias
import seaborn as sns
import matplotlib.pyplot as plt


# Generate scatterplot using support and confidence
sns.scatterplot(x = "support", y = "confidence", 
                size = "lift", data = rules)
plt.show()
print(end_time-start_time)

# %%



