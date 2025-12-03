import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

pd.set_option('future.no_silent_downcasting', True)
#======================== PART 1 ==========================
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
plt.close()
# ====== One-Hot Encoding ======
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_data = encoder.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(encoded_data,columns=encoder.get_feature_names_out(cat_cols))
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
#======================== PART 2 ==========================
X = df.drop("num", axis=1)
y = df["num"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# ----- Baseline Model: Logistic Regression -----
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
# ----- Metrics -----
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted")
rec = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
auc = roc_auc_score(y_test, y_prob, multi_class="ovr")

print(f"Baseline Logistic Regression -> Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

#======================== PART 3 ==========================
configs = [("RF_50_5",   50, 5),("RF_100_10", 100, 10),("RF_200_None", 200, None)]
results = []
for name, est, depth in configs:
    rf = RandomForestClassifier(n_estimators=est,max_depth=depth,max_features="sqrt",random_state=42,n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)

    results.append([name,est,depth,accuracy_score(y_test, y_pred),f1_score(y_test, y_pred, average="weighted")])
# Table
rf_df = pd.DataFrame(results, columns=["name", "n_estimators", "max_depth", "Accuracy", "F1"])
print(rf_df)
# גרף n_estimators מול F1
plt.plot(rf_df["n_estimators"], rf_df["F1"], marker="o")
plt.xlabel("n_estimators")
plt.ylabel("F1 score")
plt.title("Random Forest: Trees vs F1")
plt.savefig("TreesVSf1.png", dpi=300)
rf_df.to_csv("RF_Results_Table.csv", index=False)
#======================== PART 4 ==========================

ada_configs = [("ADA_50_1.0",   50, 1.0),("ADA_100_0.5", 100, 0.5),("ADA_200_2.0", 200, 2.0)]
ada_results = []
for name, est, lr in ada_configs:
    ada = AdaBoostClassifier(n_estimators=est, learning_rate=lr, random_state=42)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    y_prob = ada.predict_proba(X_test)

    ada_results.append([name,est,lr,accuracy_score(y_test, y_pred),f1_score(y_test, y_pred, average="weighted"),roc_auc_score(y_test, y_prob, multi_class="ovr")])
ada_df = pd.DataFrame(ada_results, columns=["name","n_estimators","learning_rate","Accuracy","F1","AUC"])
print("\nAdaBoost Results:")
print(ada_df)
ada_df.to_csv("ADA_Results_Table.csv", index=False)
from xgboost import XGBClassifier
import time

# ======================== PART 5 ==========================
configs_xgb = [("XGB_100_3_lr0.1", 100, 3, 0.1, 1.0),("XGB_200_4_lr0.05", 200, 4, 0.05, 0.8),("XGB_300_5_lr0.1", 300, 5, 0.1, 0.7)]
xgb_results = []
for name, est, depth, lr, subs in configs_xgb:
    start = time.time()
    xgb = XGBClassifier(n_estimators=est,max_depth=depth,learning_rate=lr,subsample=subs,colsample_bytree=0.8,objective="multi:softprob",eval_metric="mlogloss",random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)
    end = time.time()
    xgb_results.append([name, est, depth, lr, subs,accuracy_score(y_test, y_pred),f1_score(y_test, y_pred, average="weighted"),roc_auc_score(y_test, y_prob, multi_class="ovr"),end - start])
xgb_df = pd.DataFrame(xgb_results,columns=["name", "n_estimators", "max_depth", "lr", "subsample", "Accuracy", "F1", "AUC", "Train_time"])
print("\nXGBoost Results:")
print(xgb_df)
xgb_df.to_csv("XGBoost_Results_Table.csv", index=False)

