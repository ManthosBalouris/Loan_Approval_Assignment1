import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
 

df = pd.read_csv('loan_approval_dataset.csv', sep=';', skipinitialspace=True)
df.columns = df.columns.str.strip()
for c in df.select_dtypes(exclude='number'):
    df[c] = df[c].str.strip()
 
print(df.head(20))
print(df.shape, df.isna().sum().sum(), 'missing values')
print(df.describe().T)
print(df['loan_status'].value_counts(normalize=True))
 

df['loan_status_num'] = (df['loan_status'] == 'Approved').astype(int)
numeric = df.drop(columns=['loan_id']).select_dtypes(include='number')
corrmatrix = numeric.corr()
print(corrmatrix['loan_status_num'].sort_values())
 
plt.figure(figsize=(10, 8))
sns.heatmap(corrmatrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation heat map (numeric features)')
plt.tight_layout()
plt.savefig('heatmap.png', dpi=150)
plt.show()
 

for col in ['education', 'self_employed']:
    ct = pd.crosstab(df[col], df['loan_status'])
    print(ct)
    print(pd.crosstab(df[col], df['loan_status'], normalize='index').round(3))
    chi2, p, _, _ = stats.chi2_contingency(ct)
    print(f'{col}: chi2={chi2:.3f}, p={p:.3f}')
 
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(data=df, x='education', hue='loan_status', ax=ax[0])
ax[0].set_title('Education vs Loan_Status')
sns.boxplot(data=df, x='loan_status', y='cibil_score', ax=ax[1])
ax[1].set_title('CIBIL score by Loan_Status')
plt.tight_layout()
plt.savefig('categorical.png', dpi=150)
plt.show()
