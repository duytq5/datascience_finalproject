"""
Comprehensive Missing Data Analysis and Imputation for Titanic Dataset

This script demonstrates multiple imputation techniques with detailed explanations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# STEP 1: LOAD DATA AND ANALYZE MISSING PATTERNS
# ============================================================================

print("=" * 80)
print("MISSING DATA ANALYSIS - TITANIC DATASET")
print("=" * 80)

df = pd.read_csv('titanic_data.csv')

print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nColumn types:\n{df.dtypes}")

print("\n2. MISSING VALUES SUMMARY")
print("-" * 80)
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
print(missing_data.to_string(index=False))

print("\n3. MISSING DATA PATTERNS")
print("-" * 80)
print(f"Rows with any missing value: {df.isnull().any(axis=1).sum()} ({df.isnull().any(axis=1).sum()/len(df)*100:.1f}%)")
print(f"Rows with ALL values present: {(~df.isnull().any(axis=1)).sum()} ({(~df.isnull().any(axis=1)).sum()/len(df)*100:.1f}%)")


# ============================================================================
# STEP 2: VISUALIZE MISSING DATA
# ============================================================================

print("\n4. CREATING VISUALIZATIONS...")
print("-" * 80)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Missing Data Analysis - Titanic Dataset', fontsize=16, fontweight='bold')

# Plot 1: Missing data heatmap
ax1 = axes[0, 0]
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax1)
ax1.set_title('Missing Data Pattern (Yellow = Missing)', fontweight='bold')
ax1.set_xlabel('Columns')

# Plot 2: Missing data bar chart
ax2 = axes[0, 1]
missing_counts = df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
missing_counts.plot(kind='bar', ax=ax2, color='coral')
ax2.set_title('Missing Values Count by Column', fontweight='bold')
ax2.set_ylabel('Number of Missing Values')
ax2.set_xlabel('Column')
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Age distribution (with/without missing)
ax3 = axes[1, 0]
df['Age'].hist(bins=20, ax=ax3, color='skyblue', edgecolor='black', alpha=0.7)
ax3.set_title(f'Age Distribution (n={df["Age"].notna().sum()}, missing={df["Age"].isna().sum()})', fontweight='bold')
ax3.set_xlabel('Age')
ax3.set_ylabel('Frequency')

# Plot 4: Missing Age by Pclass and Sex
ax4 = axes[1, 1]
age_missing = df.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.isna().sum())
age_missing.unstack().plot(kind='bar', ax=ax4, color=['lightcoral', 'lightblue'])
ax4.set_title('Missing Age by Passenger Class and Sex', fontweight='bold')
ax4.set_ylabel('Count of Missing Age')
ax4.set_xlabel('Passenger Class')
ax4.legend(title='Sex')
ax4.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('missing_data_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to 'missing_data_visualization.png'")


# ============================================================================
# STEP 3: IMPUTATION METHODS COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("IMPUTATION METHODS DEMONSTRATION")
print("=" * 80)

# Create copies for different imputation methods
df_original = df.copy()

# ============================================================================
# METHOD 1: SIMPLE MEAN/MEDIAN IMPUTATION
# ============================================================================

print("\n5. METHOD 1: SIMPLE MEDIAN IMPUTATION")
print("-" * 80)

df_median = df.copy()

# Age: Fill with overall median
age_median = df_median['Age'].median()
df_median['Age'].fillna(age_median, inplace=True)
print(f"✓ Age: Filled {df['Age'].isna().sum()} missing values with median = {age_median:.2f}")

# Embarked: Fill with mode
embarked_mode = df_median['Embarked'].mode()[0]
df_median['Embarked'].fillna(embarked_mode, inplace=True)
print(f"✓ Embarked: Filled {df['Embarked'].isna().sum()} missing values with mode = '{embarked_mode}'")

# Cabin: Fill with 'Unknown'
df_median['Cabin'].fillna('Unknown', inplace=True)
print(f"✓ Cabin: Filled {df['Cabin'].isna().sum()} missing values with 'Unknown'")

print(f"\nRemaining missing values: {df_median.isnull().sum().sum()}")


# ============================================================================
# METHOD 2: GROUP-BASED IMPUTATION (RECOMMENDED)
# ============================================================================

print("\n6. METHOD 2: GROUP-BASED MEDIAN IMPUTATION (RECOMMENDED)")
print("-" * 80)

df_group = df.copy()

# Age: Fill based on Pclass and Sex groups
for pclass in df_group['Pclass'].unique():
    for sex in df_group['Sex'].unique():
        mask = (df_group['Pclass'] == pclass) & (df_group['Sex'] == sex) & (df_group['Age'].isna())
        group_median = df_group[(df_group['Pclass'] == pclass) & (df_group['Sex'] == sex)]['Age'].median()
        df_group.loc[mask, 'Age'] = group_median
        if mask.sum() > 0:
            print(f"  Pclass={pclass}, Sex={sex}: Filled {mask.sum()} values with median={group_median:.2f}")

# Embarked: Mode
df_group['Embarked'].fillna(df_group['Embarked'].mode()[0], inplace=True)

# Cabin: Create binary feature
df_group['HasCabin'] = df_group['Cabin'].notna().astype(int)
print(f"\n✓ Cabin: Created 'HasCabin' binary feature (1={df_group['HasCabin'].sum()}, 0={len(df_group)-df_group['HasCabin'].sum()})")

print(f"\nRemaining missing values: {df_group.isnull().sum().sum()}")


# ============================================================================
# METHOD 3: TITLE-BASED IMPUTATION (DOMAIN-SPECIFIC)
# ============================================================================

print("\n7. METHOD 3: TITLE-BASED IMPUTATION (DOMAIN-SPECIFIC)")
print("-" * 80)

df_title = df.copy()

# Extract title from name
df_title['Title'] = df_title['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print(f"Extracted titles: {df_title['Title'].value_counts().to_dict()}")

# Fill age based on title
print("\nAge imputation by title:")
for title in df_title['Title'].unique():
    mask = (df_title['Title'] == title) & (df_title['Age'].isna())
    title_median = df_title[df_title['Title'] == title]['Age'].median()
    df_title.loc[mask, 'Age'] = title_median
    if mask.sum() > 0:
        print(f"  {title}: Filled {mask.sum()} values with median={title_median:.2f}")

# Embarked and Cabin
df_title['Embarked'].fillna(df_title['Embarked'].mode()[0], inplace=True)
df_title['HasCabin'] = df_title['Cabin'].notna().astype(int)

print(f"\nRemaining missing values: {df_title.isnull().sum().sum()}")


# ============================================================================
# METHOD 4: KNN IMPUTATION
# ============================================================================

print("\n8. METHOD 4: KNN IMPUTATION (MACHINE LEARNING)")
print("-" * 80)

df_knn = df.copy()

# Prepare numeric features for KNN
features_for_knn = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
df_knn['Sex_encoded'] = (df_knn['Sex'] == 'male').astype(int)
features_for_knn.append('Sex_encoded')

# KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
df_knn[features_for_knn] = knn_imputer.fit_transform(df_knn[features_for_knn])

print(f"✓ Age: Filled using KNN with 5 nearest neighbors")
print(f"✓ Features used: {features_for_knn}")

# Handle other columns
df_knn['Embarked'].fillna(df_knn['Embarked'].mode()[0], inplace=True)
df_knn['HasCabin'] = df_knn['Cabin'].notna().astype(int)

print(f"\nRemaining missing values in numeric columns: {df_knn[features_for_knn].isnull().sum().sum()}")


# ============================================================================
# METHOD 5: PREDICTIVE IMPUTATION (RANDOM FOREST)
# ============================================================================

print("\n9. METHOD 5: PREDICTIVE IMPUTATION (RANDOM FOREST)")
print("-" * 80)

df_rf = df.copy()

# Prepare features
df_rf['Sex_encoded'] = (df_rf['Sex'] == 'male').astype(int)
feature_cols = ['Pclass', 'Sex_encoded', 'SibSp', 'Parch', 'Fare']

# Separate data with/without age
df_with_age = df_rf[df_rf['Age'].notna()]
df_without_age = df_rf[df_rf['Age'].isna()]

if len(df_without_age) > 0:
    # Train Random Forest on known ages
    X_train = df_with_age[feature_cols]
    y_train = df_with_age['Age']

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)

    # Predict missing ages
    X_predict = df_without_age[feature_cols]
    predicted_ages = rf_model.predict(X_predict)

    df_rf.loc[df_rf['Age'].isna(), 'Age'] = predicted_ages

    print(f"✓ Age: Filled {len(df_without_age)} values using Random Forest")
    print(f"  Feature importance: {dict(zip(feature_cols, rf_model.feature_importances_.round(3)))}")

# Handle other columns
df_rf['Embarked'].fillna(df_rf['Embarked'].mode()[0], inplace=True)
df_rf['HasCabin'] = df_rf['Cabin'].notna().astype(int)

print(f"\nRemaining missing values: {df_rf.isnull().sum().sum()}")


# ============================================================================
# STEP 4: COMPARE IMPUTATION RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("COMPARISON OF IMPUTATION METHODS")
print("=" * 80)

# Compare Age distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Age Distribution Comparison Across Imputation Methods', fontsize=16, fontweight='bold')

methods = [
    ('Original (with missing)', df_original['Age'].dropna(), axes[0, 0]),
    ('Method 1: Simple Median', df_median['Age'], axes[0, 1]),
    ('Method 2: Group Median', df_group['Age'], axes[0, 2]),
    ('Method 3: Title-based', df_title['Age'], axes[1, 0]),
    ('Method 4: KNN', df_knn['Age'], axes[1, 1]),
    ('Method 5: Random Forest', df_rf['Age'], axes[1, 2])
]

for method_name, age_data, ax in methods:
    ax.hist(age_data, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(method_name, fontweight='bold')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    ax.axvline(age_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={age_data.mean():.1f}')
    ax.axvline(age_data.median(), color='orange', linestyle='--', linewidth=2, label=f'Median={age_data.median():.1f}')
    ax.legend()

plt.tight_layout()
plt.savefig('imputation_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved comparison visualization to 'imputation_comparison.png'")


# ============================================================================
# STEP 5: SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

summary = pd.DataFrame({
    'Method': ['Original', 'Simple Median', 'Group Median', 'Title-based', 'KNN', 'Random Forest'],
    'Mean Age': [
        df_original['Age'].mean(),
        df_median['Age'].mean(),
        df_group['Age'].mean(),
        df_title['Age'].mean(),
        df_knn['Age'].mean(),
        df_rf['Age'].mean()
    ],
    'Median Age': [
        df_original['Age'].median(),
        df_median['Age'].median(),
        df_group['Age'].median(),
        df_title['Age'].median(),
        df_knn['Age'].median(),
        df_rf['Age'].median()
    ],
    'Std Dev': [
        df_original['Age'].std(),
        df_median['Age'].std(),
        df_group['Age'].std(),
        df_title['Age'].std(),
        df_knn['Age'].std(),
        df_rf['Age'].std()
    ]
})

print(summary.to_string(index=False))

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print("""
Based on the analysis:

1. **BEST OVERALL**: Method 3 (Title-based) or Method 2 (Group-based)
   - Preserves domain knowledge (titles correlate with age)
   - Maintains realistic age distributions
   - Simple to implement and explain

2. **MOST ACCURATE**: Method 5 (Random Forest)
   - Uses relationships between multiple features
   - Best for predictive modeling
   - More complex but captures nuances

3. **CABIN HANDLING**: Convert to binary 'HasCabin' feature
   - 77% missing is too high for imputation
   - Presence/absence is informative (class indicator)

4. **EMBARKED**: Simple mode imputation
   - Only 1-2 missing values
   - Low impact on analysis

RECOMMENDED WORKFLOW:
- Use Title-based or Group-based imputation for Age
- Create HasCabin binary feature
- Use mode for Embarked
- Always split data BEFORE imputation to avoid data leakage
""")

print("\n✓ Analysis complete! Check the generated PNG files for visualizations.")
