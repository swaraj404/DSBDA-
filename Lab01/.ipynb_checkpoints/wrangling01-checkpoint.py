# Import pandas library
import pandas as pd

# Step 1-3: Load the CSV file
print("Loading CSV file...")
df = pd.read_csv('/home/swaraj/COLLEGE/DSBDA/Lab01/iris.csv')
print("File loaded successfully!")
print()

# STEP 4: DATA PREPROCESSING
print("=" * 50)
print("STEP 4: DATA PREPROCESSING")
print("=" * 50)

# Check dimensions
print("\n1. Dimensions of dataset:")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Check for missing values
print("\n2. Missing values in each column:")
print(df.isnull().sum())

# Use describe() function
print("\n3. Statistical summary using describe():")
print(df.describe())

# Check data types
print("\n4. Data types of each column:")
print(df.dtypes)

# STEP 5: DATA FORMATTING AND NORMALIZATION
print("\n" + "=" * 50)
print("STEP 5: DATA FORMATTING AND NORMALIZATION")
print("=" * 50)

# Fill missing values with median (if any exist)
print("\nFilling missing values...")
df['sepal_length'] = df['sepal_length'].fillna(df['sepal_length'].median())
df['sepal_width'] = df['sepal_width'].fillna(df['sepal_width'].median())
df['petal_length'] = df['petal_length'].fillna(df['petal_length'].median())
df['petal_width'] = df['petal_width'].fillna(df['petal_width'].median())
print("Missing values filled!")

# Normalize all numerical columns (between 0 and 1)
print("\nNormalizing numerical columns...")
df['Normalized_sepal_length'] = (df['sepal_length'] - df['sepal_length'].min()) / (df['sepal_length'].max() - df['sepal_length'].min())
df['Normalized_sepal_width'] = (df['sepal_width'] - df['sepal_width'].min()) / (df['sepal_width'].max() - df['sepal_width'].min())
df['Normalized_petal_length'] = (df['petal_length'] - df['petal_length'].min()) / (df['petal_length'].max() - df['petal_length'].min())
df['Normalized_petal_width'] = (df['petal_width'] - df['petal_width'].min()) / (df['petal_width'].max() - df['petal_width'].min())
print("Normalization done!")

# STEP 6: TURN CATEGORICAL TO QUANTITATIVE
print("\n" + "=" * 50)
print("STEP 6: CATEGORICAL TO QUANTITATIVE")
print("=" * 50)

# Convert Species to numbers: setosa=0, versicolor=1, virginica=2
print("\nConverting Species to numbers...")
df['Species_Numeric'] = df['species'].replace({
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
})
print("setosa = 0, versicolor = 1, virginica = 2")

# Create size category based on petal length
print("\nCreating size categories based on petal length...")
for i in range(len(df)):
    petal_len = df.loc[i, 'petal_length']
    if petal_len >= 5.0:
        df.loc[i, 'Size_Category'] = 'Large'
    elif petal_len >= 3.0:
        df.loc[i, 'Size_Category'] = 'Medium'
    else:
        df.loc[i, 'Size_Category'] = 'Small'

# Convert size category to numbers
df['Size_Category_Numeric'] = df['Size_Category'].replace({
    'Large': 2,
    'Medium': 1,
    'Small': 0
})
print("Size categories created!")

# FINAL SUMMARY
print("\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)
print("\nFirst 5 rows of processed data:")
print(df[['species', 'sepal_length', 'petal_length', 'Species_Numeric', 'Size_Category']].head())

print("\nFinal shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())

# Save to CSV
df.to_csv('iris_Processed.csv', index=False)
print("\nProcessed data saved to 'iris_Processed.csv'")
print("\nDone!")
