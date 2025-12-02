import pandas as pd
import seaborn as sns

# -----------------------------------
# LOAD OPEN-SOURCE DATA (no file path)
# -----------------------------------
df = sns.load_dataset("diabetes")

# -----------------------------------
# BASIC DATA CLEANING
# -----------------------------------

# 1. Remove duplicate rows
df = df.drop_duplicates()

# 2. Handle missing values
# Strategy:
# - Numeric columns → fill with median
# - Categorical columns → fill with mode
for col in df.columns:
    if df[col].dtype != "object":
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# 3. Fix column names (lowercase, replace spaces)
df.columns = df.columns.str.lower().str.replace(" ", "_")

# 4. Remove outliers using IQR method
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in numeric_cols:
    df = remove_outliers_iqr(df, col)

# 5. Optional: convert categorical to category type
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].astype("category")

# -----------------------------------
# CLEANED DATA READY
# -----------------------------------
cleaned_df = df
