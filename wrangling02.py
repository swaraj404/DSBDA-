# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("Loading Demo Marks.xlsx...")
df = pd.read_excel('Demo Marks.xlsx')
print("Dataset loaded successfully!\n")

print("=" * 60)
print("TASK 1: HANDLING MISSING VALUES AND INCONSISTENCIES")
print("=" * 60)

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Fill missing numeric values with median
print("\nFilling missing values with median...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"Filled {col}")

print("\nMissing values after filling:")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("TASK 2: HANDLING OUTLIERS")
print("=" * 60)

# Method 1: Boxplot
print("\nMethod 1: Boxplot")
df[numeric_cols].boxplot(figsize=(12, 6))
plt.savefig('outliers_boxplot.png')
plt.show()

# Method 2: Scatter plot
print("Method 2: Scatter plot")
if len(numeric_cols) >= 2:
    plt.figure(figsize=(10, 6))
    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.savefig('outliers_scatter.png')
    plt.show()

# Method 3: Z-Score
print("Method 3: Z-Score")
from scipy import stats
for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df[col]))
    outliers = df[z_scores > 3]
    if len(outliers) > 0:
        print(f"  {col}: {len(outliers)} outliers")

# Method 4: IQR method (treating outliers)
print("Method 4: IQR method")
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"  {col}: IQR = {IQR:.2f}, Q1 = {Q1:.2f}, Q3 = {Q3:.2f}")
    if len(outliers) > 0:
        print(f"    {len(outliers)} outliers - capped to [{lower:.2f}, {upper:.2f}]")
        df[col] = df[col].clip(lower, upper)

print("\n" + "=" * 60)
print("TASK 3: DATA TRANSFORMATION")
print("=" * 60)

# Apply log transformation to reduce skewness
print("\nApplying log transformation to 'Total 100%' column...")
print("Reason: To reduce skewness and normalize the distribution")

# Check skewness before transformation
if 'Total 100%' in df.columns:
    original_skewness = df['Total 100%'].skew()
    print(f"\nOriginal skewness: {original_skewness:.4f}")
    
    # Apply log transformation (add 1 to avoid log(0))
    df['Log_Total'] = np.log1p(df['Total 100%'])
    new_skewness = df['Log_Total'].skew()
    print(f"Skewness after log transformation: {new_skewness:.4f}")
    print(f"Skewness reduced by: {abs(original_skewness - new_skewness):.4f}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nFinal dataset shape: {df.shape}")
print(f"Total missing values: {df.isnull().sum().sum()}")
print("\nFirst 5 rows:")
print(df.head())

# Save processed data
df.to_csv('Academic_Performance_Processed.csv', index=False)
print("\n✓ Processed data saved to 'Academic_Performance_Processed.csv'")
print("\nDone!")
