import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

pd.set_option('future.no_silent_downcasting', True)

# ----- Load the dataset -----
print("Dataset Table (5 rows):")
df = pd.read_csv("heart_disease_uci.csv")
print(df.head())
# Deleting irrelevant columns
df = df.drop(columns=["id", "dataset"])   # כאן מוחקים id ו-dataset מהדאטה
# ----- Check for missing values -----
print("\nMissing values per column:")
print(df.isnull().sum())

# ----- Class imbalance -----
print("\nTarget distribution (num) - Class imbalance:")
print(df["num"].value_counts())

# ----- Handle Missing Values -----
# Numeric columns → fill with median
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical columns → fill with mode
cat_cols = df.select_dtypes(include=["object"]).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

print("\nMissing values after filling:")
print(df.isnull().sum())

# ----- Correlation Heatmap -----
numeric_df = df.select_dtypes(include=["int64", "float64"])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")

# ====== One-Hot Encoding ======
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_data = encoder.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(cat_cols)
)
df = df.drop(columns=cat_cols)

# add new columns
df = pd.concat([df, encoded_df], axis=1)

print("\nAfter One-Hot Encoding:")
print(df.head())
print(df.shape)
print(df.columns)

from sklearn.preprocessing import StandardScaler

# ----- Standardization -----
scaler = StandardScaler()
feature_cols = [col for col in df.columns if col != "num"]
df[feature_cols] = scaler.fit_transform(df[feature_cols])

print("\nAfter Scaling:")
print(df.head())
