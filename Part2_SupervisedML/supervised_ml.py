import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load Clean Dataset
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Clean TotalCharges and customerID just like Part 1
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

# 2. Define Features and Targets
# --- Classification Target: Churn (Yes=1, No=0) ---
y_class = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# --- Regression Target: TotalCharges (continuous prediction) ---
y_reg = df['TotalCharges']

# Drop targets from feature matrix X
X = df.drop(columns=['Churn', 'TotalCharges'])

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

print(f"Numeric Features: {numeric_features}")
print(f"Categorical Features: {categorical_features}")

# 3. Preprocessing Transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 4. Train-Test Splits
# Classification Split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)

# Regression Split (using MonthlyCharges as feature, dropping TotalCharges from X already done)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# ==========================================
# 5. SUPERVISED CLASSIFICATION MODELS
# ==========================================
print("\n--- Training Classification Models ---")
clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42)
}

for name, model in clf_models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipe.fit(X_train_c, y_train_c)
    y_pred = pipe.predict(X_test_c)
    y_prob = pipe.predict_proba(X_test_c)[:, 1] if hasattr(pipe, "predict_proba") else y_pred
    
    print(f"\nModel: {name}")
    print(f"  Accuracy:  {accuracy_score(y_test_c, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test_c, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test_c, y_pred):.4f}")
    print(f"  F1-Score:  {f1_score(y_test_c, y_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test_c, y_prob):.4f}")

# ==========================================
# 6. SUPERVISED REGRESSION MODELS
# ==========================================
print("\n--- Training Regression Models ---")
reg_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
}

for name, model in reg_models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipe.fit(X_train_r, y_train_r)
    y_pred = pipe.predict(X_test_r)
    
    print(f"\nModel: {name}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred)):.4f}")
    print(f"  MAE:  {mean_absolute_error(y_test_r, y_pred):.4f}")
    print(f"  R2:   {r2_score(y_test_r, y_pred):.4f}")