import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#This is doen so that I know which columns have strong correlations with others (looking for wins / winrate correlations)


df = pd.read_csv("DraftStats.csv")

# clean up and convert numeric columns
df.columns = df.columns.str.strip()

# try converting all numeric-looking columns to float
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# drop text columns (Champion, Role)
num_df = df.drop(columns=["Champion", "Role"], errors="ignore")

# compute correlation matrix
corr = num_df.corr()

# display as text summary (sorted by target correlation)
target = "Winrate"
if target in corr.columns:
    print(f"\nTop correlations with {target}:")
    print(corr[target].sort_values(ascending=False))
else:
    print("'Winrate' column not found in numeric data.")

# visualize
plt.figure(figsize=(10, 8))
plt.title("Champion Stats Correlation Matrix", fontsize=14)
plt.imshow(corr, cmap="coolwarm", interpolation="none")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.show()