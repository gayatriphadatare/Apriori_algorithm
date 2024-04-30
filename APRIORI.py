#!/usr/bin/env python
# coding: utf-8
Web Usage Mining: Use the Apriori algorithm to analyze web log data and discover frequent
navigation patterns or sequences of web pages visited by users. 

user id                visited pages
1                 {home, products, about us}
2                 {home, contact us}
3                 {home, product, blog}
4                 {home, blog}
5                 {home, products, contact us}

# In[8]:


pip install mlxtend


# In[7]:


# Importing required libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# Web log data
data = [
    ['home', 'products', 'about us'],
    ['home', 'contact us'],
    ['home', 'product', 'blog'],
    ['home', 'blog'],
    ['home', 'products', 'contact us']
]

# Transforming data for Apriori algorithm
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)

# Converting transformed data into DataFrame
df = pd.DataFrame(te_ary, columns=te.columns_)

# Applying Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# Printing the result
print("Frequent Itemsets:")
print(frequent_itemsets)

