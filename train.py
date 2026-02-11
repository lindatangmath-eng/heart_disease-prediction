import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# =========================
# 1. Load Dataset
# =========================
# Downloading the dataset as seen in the notebook
df = pd.read_csv("D:\\heart_disease-classification\\heart_disease.csv")

# Preprocessing: Drop 'id' and 'dataset' as they aren't predictive features
if 'id' in df.columns:
    df = df.drop(columns=['id'])
if 'dataset' in df.columns:
    df = df.drop(columns=['dataset'])

# Drop missing values
df = df.dropna()

# =========================
# 2. Define Target
# =========================
# The notebook uses "num" where 0 is healthy and >0 indicates disease
if "num" in df.columns:
    y = df["num"].apply(lambda x: 1 if x > 0 else 0)  # Binary classification
    X = df.drop("num", axis=1)
else:
    raise ValueError("Target column 'num' not found. Check your dataset.")

# =========================
# 3. Identify Column Types
# =========================
# Identify numeric and categorical features for the ColumnTransformer
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object", "category", "bool"]).columns

# =========================
# 4. Preprocessing
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

# =========================
# 5. Pipeline
# =========================
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# =========================
# 6. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 7. Train Model
# =========================
pipeline.fit(X_train, y_train)

# =========================
# 8. Evaluate Model
# =========================
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# 9. Save Model
# =========================
with open("heart_disease_pipeline.pkl", "wb") as file:
    pickle.dump(pipeline, file)

print("Model saved successfully as 'heart_disease_pipeline.pkl'!")