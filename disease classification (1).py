#!/usr/bin/env python
# coding: utf-8

# ## 1. Data collection

# In[1]:


get_ipython().system('pip install GEOparse')


# In[2]:


import GEOparse


# In[3]:


gse=GEOparse.get_GEO(filepath="C:\\Users\\sindu\\Downloads\\GSE1456_family.soft.gz")


# In[25]:


"""
🧪 Summary
-----------------------------------------
Object         | What it Contains           | Use
---------------|----------------------------|-------------------------------
gse            | Full GEO Series object     | The whole dataset
gse.metadata   | Study-level info           | Read experiment summary
gse.gsms       | Dict of GSM samples        | Loop through for sample data
gsm.metadata   | One sample’s phenotype     | Use for classification labels
gsm.table      | One sample’s expression    | Gets merged into data_matrix
               | values                     |
"""


# In[4]:


print(gse.metadata.keys())


# In[5]:


print(list(gse.gsms.keys())[:5])


# In[6]:


for gsm_name,gsm in gse.gsms.items():
    print(gsm_name, gsm.metadata["characteristics_ch1"])
    break


# In[7]:


expression_data={}




# In[8]:


labels = {}
for gsm_name,gsm in gse.gsms.items():
    expr=gsm.table.set_index('ID_REF')['VALUE']
    expression_data[gsm_name]=expr
    for field in gsm.metadata["characteristics_ch1"]:
        if "RELAPSE" in field:
            relapse_value = field.split(":")[1].strip()
            labels[gsm_name] = int(relapse_value)  # convert to integer 0 or 1
            break


# In[9]:


import pandas as pd


# In[134]:


# Step 2: Build expression matrix
X_filtered_raw = pd.DataFrame(expression_data).T  # transpose so rows = samples

# Step 3: Build label vector, ordered same as X
y = [labels[sample] for sample in X_filtered_raw.index]

# Convert to Pandas Series (optional)
y = pd.Series(y, index=X_filtered_raw.index, name="relapse")


# ## preprocessing

# In[ ]:


# 1. Drop genes (columns) missing in more than 50% of samples
missing_per_gene = X_filtered_raw.isnull().sum()
X_filtered = X_filtered_raw.loc[:, missing_per_gene < (0.5 * X_filtered_raw.shape[0])]

# 2. Impute missing values in remaining genes
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
sample_index = X_filtered.index
gene_columns = X_filtered.columns

X_filtered = pd.DataFrame(imputer.fit_transform(X_filtered),
                          index=sample_index,
                          columns=gene_columns)

# 3. Align y to the cleaned X
y_filtered = y.loc[X_filtered.index]


# In[138]:


X_filtered_raw.shape


# In[139]:


y_filtered.shape


# In[128]:


from sklearn.preprocessing import StandardScaler


# In[129]:


scale=StandardScaler()


# In[130]:


X_scaled=scale.fit_transform(X_filtered)
X_scaled = pd.DataFrame(scale.fit_transform(X_filtered),
                        index=X_filtered.index,
                        columns=X_filtered.columns)


# In[131]:


from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0.01)
X_var = vt.fit_transform(X_scaled)
X_var = pd.DataFrame(X_var, index=X_scaled.index, columns=X_scaled.columns[vt.get_support()])


# ## feature selection

# In[20]:


from sklearn.feature_selection import SelectKBest , f_classif


# In[116]:


from sklearn.feature_selection import SelectKBest, f_classif

k = int(X_var.shape[1] * 0.5)
print(k)
selector = SelectKBest(score_func=f_classif, k=k)
X_selected = selector.fit_transform(X_var, y_filtered)
X_selected = pd.DataFrame(X_selected, index=X_var.index, columns=X_var.columns[selector.get_support()])


# In[82]:


from sklearn.decomposition import PCA

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_selected)


# In[28]:


import matplotlib.pyplot as plt


# In[29]:


plt.figure(figsize=(15,7))


# In[83]:


scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=y_filtered, cmap='coolwarm', alpha=0.7
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Gene Expression Data (2 Components)")
plt.colorbar(scatter, label="Relapse Status (y)")
plt.grid(True)
plt.tight_layout()
plt.show()


# ## training the model

# In[37]:


from sklearn.ensemble import RandomForestClassifier


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


from sklearn.metrics import accuracy_score,f1_score,recall_score,confusion_matrix,classification_report


# In[84]:


x_train,x_test,y_train,y_test=train_test_split(X_selected,y_filtered,test_size=0.3,random_state=42)


# In[93]:


rf = RandomForestClassifier(class_weight="balanced", random_state=42)


# In[94]:


rf.fit(x_train,y_train)


# In[95]:


y_pred = rf.predict(x_test)


# In[96]:


accuracy_score(y_test,y_pred)


# In[100]:


recall_score(y_test,y_pred)


# In[97]:


f1_score(y_test,y_pred)


# In[98]:


cm=confusion_matrix(y_test,y_pred)


# In[99]:


cm


# In[118]:


report = classification_report(y_test, y_pred)
print(report)


# In[51]:


import seaborn as sns


# In[71]:


sns.heatmap(cm,annot=True, fmt="d", cmap="Blues")


# In[92]:


import pandas as pd
import matplotlib.pyplot as plt

# 1. Get importance scores from the trained model
importances = rf.feature_importances_
feature_names = X_selected.columns

# 2. Build a DataFrame for sorting and visualization
feat_df = pd.DataFrame({
    'Gene': feature_names,
    'Importance': importances
})

# 3. Sort and select top N genes (e.g., top 20)
top_genes = feat_df.sort_values(by="Importance", ascending=False).head(20)

# 4. Plot
plt.figure(figsize=(10, 6))
plt.barh(top_genes['Gene'], top_genes['Importance'], color='skyblue')
plt.xlabel("Feature Importance Score")
plt.title("Top 20 Most Predictive Genes")
plt.gca().invert_yaxis()  # Highest importance on top
plt.tight_layout()
plt.show()


# ## trial on synthetic data

# In[103]:


import numpy as np


# In[110]:


np.random.seed(110)
synthetic_sample = X_selected.mean() + np.random.normal(0, 0.5, size=X_selected.shape[1])

# Step 2: Reshape for prediction
synthetic_sample = synthetic_sample.values.reshape(1, -1)

# Step 3: Predict using your model
pred = rf.predict(synthetic_sample)
prob = rf.predict_proba(synthetic_sample)[0][1]

# Step 4: Output
print("Predicted class (0=no relapse, 1=relapse):", pred[0])
print("Probability of relapse:", round(prob, 2))


# In[112]:


relapse_patients = X_selected[y_filtered == 1]
template = relapse_patients.sample(random_state=42).iloc[0]


# In[115]:


np.random.seed(3)
synthetic_sample = template + np.random.normal(0, 0.3, size=template.shape)
synthetic_sample = synthetic_sample.values.reshape(1, -1)

pred = rf.predict(synthetic_sample)
prob = rf.predict_proba(synthetic_sample)[0][1]

print("Prediction:", pred[0])
print("Probability of relapse:", round(prob, 2))


# In[ ]:




