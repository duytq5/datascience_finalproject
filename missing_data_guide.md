# Comprehensive Guide to Handling Missing Data in Titanic Dataset

## Missing Data Analysis

Based on your Titanic dataset, the missing values are in:
- **Age**: ~19% missing (15 out of 81 rows shown)
- **Cabin**: ~77% missing (most passengers don't have cabin info)
- **Embarked**: 1 missing value (row 62)

---

## Detailed Recommendations by Column

### 1. AGE (Numerical - Critical for Analysis)

#### **Recommended Approaches:**

**A. Median Imputation by Group (BEST for Titanic)**
```python
# Fill missing age based on passenger class and sex
df['Age'].fillna(df.groupby(['Pclass', 'Sex'])['Age'].transform('median'), inplace=True)
```
**Why?** Age likely varies by class and gender. First-class passengers were typically older, and this preserves realistic patterns.

**B. Predictive Imputation (Most Accurate)**
```python
# Use other features to predict age
from sklearn.ensemble import RandomForestRegressor

# Features: Pclass, SibSp, Parch, Fare, Sex_encoded
age_model = RandomForestRegressor()
age_model.fit(X_train, y_train)  # Train on non-null ages
df.loc[df['Age'].isnull(), 'Age'] = age_model.predict(X_missing)
```
**Why?** Uses relationships between features (fare, class, family size) to predict age more accurately.

**C. Title-Based Imputation (Domain-Specific)**
```python
# Extract title from name (Mr, Mrs, Miss, Master)
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Age'].fillna(df.groupby('Title')['Age'].transform('median'), inplace=True)
```
**Why?** Titles like "Master" (boys), "Miss" (unmarried women) correlate strongly with age ranges.

---

### 2. CABIN (Categorical - High Missing Rate)

#### **Recommended Approaches:**

**A. Create Binary Feature (BEST for Predictive Modeling)**
```python
# Convert to "Has Cabin" vs "No Cabin"
df['HasCabin'] = df['Cabin'].notna().astype(int)
df.drop('Cabin', axis=1, inplace=True)
```
**Why?** With 77% missing, imputation is unreliable. Whether someone had cabin info recorded may indicate ticket class or survival (better cabins = higher class).

**B. Fill with "Unknown" Category**
```python
df['Cabin'].fillna('Unknown', inplace=True)
```
**Why?** Keeps the column if you need cabin deck analysis (first letter indicates deck: A, B, C, etc.).

**C. Extract Deck Information**
```python
df['Deck'] = df['Cabin'].str[0]  # First letter = deck
df['Deck'].fillna('Unknown', inplace=True)
df.drop('Cabin', axis=1, inplace=True)
```
**Why?** Deck location might correlate with survival (upper decks = easier escape).

---

### 3. EMBARKED (Categorical - Minimal Missing)

#### **Recommended Approaches:**

**A. Mode Imputation (BEST for Few Missing Values)**
```python
# Fill with most common port (likely 'S' - Southampton)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```
**Why?** Only 1 missing value out of 81 shown. Mode (most frequent) is safest assumption.

**B. Fare-Based Imputation**
```python
# Find similar passengers by fare and class
median_fare = df[df['Embarked'].notna()].groupby(['Pclass', 'Embarked'])['Fare'].median()
# Match missing row's fare to most likely port
```
**Why?** Ticket fare varied by embarkation port. More sophisticated but overkill for 1 value.

---

## Strategy Selection Guide

### Choose Based on Your Goal:

| Goal | Age Strategy | Cabin Strategy | Embarked Strategy |
|------|-------------|----------------|-------------------|
| **Quick EDA** | Median by Pclass | Drop column | Mode |
| **Machine Learning** | Predictive/Title-based | HasCabin binary | Mode |
| **Statistical Analysis** | Multiple Imputation (MICE) | Deck extraction | Mode |
| **Maximum Accuracy** | Random Forest imputation | HasCabin + Deck | Fare-based |

---

## Advanced Techniques

### 1. Multiple Imputation (MICE)
Best for statistical inference when you need uncertainty estimates.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=0)
df_imputed = imputer.fit_transform(df[numeric_columns])
```

**Pros:**
- Accounts for uncertainty in missing values
- Uses relationships between all variables
- Provides multiple complete datasets

**Cons:**
- Computationally expensive
- Requires all features to be numeric (need encoding first)

---

### 2. KNN Imputation
Fills missing values using K-nearest neighbors.

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df[numeric_columns])
```

**Best for:** Age (finds similar passengers by class, fare, family size)

---

### 3. Forward/Backward Fill
Only for time-series data (NOT applicable to Titanic).

```python
df['Age'].fillna(method='ffill')  # Use previous valid value
```

---

## Evaluation Methods

### How to Choose the Best Imputation?

1. **Cross-Validation**: Compare model performance with different imputation methods
2. **Distribution Check**: Ensure imputed values match original distribution
3. **Domain Knowledge**: Age of "Master" should be < 18, etc.

```python
# Compare distributions
import matplotlib.pyplot as plt

# Before imputation
original_age = df['Age'].dropna()

# After imputation (try different methods)
imputed_age = df_imputed['Age']

# Plot both
plt.hist(original_age, alpha=0.5, label='Original')
plt.hist(imputed_age, alpha=0.5, label='Imputed')
plt.legend()
```

---

## Common Mistakes to Avoid

1. **Don't impute before train-test split**
   ```python
   # WRONG: Impute entire dataset
   df['Age'].fillna(df['Age'].median())
   train, test = train_test_split(df)

   # RIGHT: Impute train and test separately
   train, test = train_test_split(df)
   median_age = train['Age'].median()
   train['Age'].fillna(median_age, inplace=True)
   test['Age'].fillna(median_age, inplace=True)  # Use training median
   ```

2. **Don't use mean for skewed distributions** (use median instead)

3. **Don't ignore why data is missing** (Cabin missing might be informative!)

4. **Don't drop rows if >10% missing** (you'd lose too much data)

---

## Recommended Workflow for Titanic Dataset

```python
# 1. Age: Title-based imputation (domain-specific)
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Age'].fillna(df.groupby('Title')['Age'].transform('median'), inplace=True)

# 2. Cabin: Convert to binary feature
df['HasCabin'] = df['Cabin'].notna().astype(int)
df.drop('Cabin', axis=1, inplace=True)

# 3. Embarked: Mode imputation
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 4. Verify no missing values remain
print(df.isnull().sum())
```

---

## Summary Table

| Column | Missing % | Best Method | Rationale |
|--------|-----------|-------------|-----------|
| Age | ~19% | Title-based median | Titles (Mr, Mrs, Master) correlate with age |
| Cabin | ~77% | Binary HasCabin | Too many missing; presence/absence is informative |
| Embarked | ~1% | Mode | Only 1 missing; use most common port |

---

## Next Steps

1. Run `missing_data_analysis.py` to visualize missing patterns
2. Experiment with different imputation methods
3. Compare model performance (e.g., survival prediction accuracy)
4. Document which method you chose and why
