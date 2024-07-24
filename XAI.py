# %% [markdown]
# ### Task 1: Import Libraries

# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import shap
shap.initjs()

# %% [markdown]
# ### Task 2: Preparing the Dataset

# %% [markdown]
# Load the dataset

# %%
df = pd.read_csv("/usercode/Census_Income.csv")
df.head()

# %% [markdown]
# Standardize the numeric columns

# %%
numeric_cols = df.select_dtypes(include=['number'])
scaler = StandardScaler()
df[numeric_cols.columns] = scaler.fit_transform(numeric_cols)
df.head()

# %% [markdown]
# One-hot-encode the categorical features

# %%
cat_cols = [1,2,4,5,6,7,8,10]
df_encoded = pd.get_dummies(df, columns=df.columns[cat_cols]) 
df.head()

# %% [markdown]
# Encode the target feature

# %%
le = LabelEncoder()
df_encoded['INCOME'] = le.fit_transform(df['INCOME'])
df_encoded.head()

# %% [markdown]
# Split the dataset into `X` and `y`

# %%
y = df_encoded['INCOME']
df_encoded.drop(columns=['INCOME'],inplace=True)
X = df_encoded

# %% [markdown]
# ### Task 3: Explain a Logistic Regression Model

# %% [markdown]
# Instantiate and fit the `LogisticRegression()` model

# %%
LR_model = LogisticRegression(C=0.205, max_iter=300, random_state=42, solver='newton-cg').fit(X,y)

# %% [markdown]
# Extract model coefficients

# %%
coefficients = LR_model.coef_[0]

# %% [markdown]
# Plot the top ten coefficients

# %%
feature_names = list(X.columns)
coef_feature_pairs = list(zip(coefficients, feature_names))
sorted_coef_feature_pairs = sorted(coef_feature_pairs, key=lambda x: abs(x[0]), 
reverse=True)
sorted_coefficients, sorted_feature_names = zip(*sorted_coef_feature_pairs)
top_ten_coefficients = sorted_coefficients[:10]
top_ten_feature_names = sorted_feature_names[:10]

plt.figure(figsize=(10, 6))
plt.barh(top_ten_feature_names, top_ten_coefficients)
plt.xlabel('Coefficient Value')
plt.ylabel('Feature Name')
plt.title('Top Ten Logistic Regression Coefficients (Sorted by Absolute Value)')
plt.gca().invert_yaxis() 
plt.show()

# %% [markdown]
# ### Task 4: Explain Random Forest Model

# %% [markdown]
# Instantiate and fit the `RandomForest()` model

# %%
RF_model = RandomForestClassifier(n_estimators=100, max_depth=40, max_features='sqrt', min_samples_split=2, min_samples_leaf=2, criterion='entropy', class_weight=None, bootstrap=True).fit(X,y)

# %% [markdown]
# Extract the feature importances

# %%
importances = RF_model.feature_importances_

# %% [markdown]
# Plot the feature importances

# %%
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', 
ascending=False)
top_n = 10   

plt.figure(figsize=(12, 6))
plt.barh(range(top_n), feature_importance_df['Importance'][:top_n], align='center')
plt.yticks(range(top_n), feature_importance_df['Feature'][:top_n])
plt.xlabel('Feature Importance')
plt.title('Top {} Feature Importances'.format(top_n))
plt.gca().invert_yaxis()  
plt.show()

# %% [markdown]
# ### Task 5: Explain Neural Network Model

# %% [markdown]
# Split the dataset into `X` and `y`

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# Instantiate and fit the `MLPClassifier()` model

# %%
NN_model = MLPClassifier(alpha=0.06, hidden_layer_sizes=(50, 50), learning_rate_init=0.03, max_iter=158).fit(X_train, y_train)


# %% [markdown]
# Calculate the permutation importances

# %%
perm_importance = permutation_importance(NN_model, X_test, y_test, n_repeats=10, random_state=42)

# %% [markdown]
# Plot the feature importances

# %%
feature_importances = perm_importance.importances_mean
sorted_idx = feature_importances.argsort()[::-1]
feature_names = X.columns
top_n = 10

plt.figure(figsize=(10, 6))
plt.bar(range(top_n), feature_importances[sorted_idx[:top_n]], align="edge")
plt.xticks(range(top_n), [feature_names[i] for i in sorted_idx[:top_n]], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Top Feature Importances")
plt.show()

# %% [markdown]
# ### Task 6: SHAP for Local Explanations

# %% [markdown]
# Create shap Explainer instance 

# %%
explainer = shap.Explainer(RF_model)

# %% [markdown]
# Compute shap values on entire dataset

# %%
shap_values = explainer(X[:100])

# %% [markdown]
# Select individual row and predict the output

# %%
RF_model.predict_proba(X)[0]

# %% [markdown]
# Plot shap values to explain the selected individual

# %%
shap.plots.waterfall(shap_values[0,:,0])

# %%
