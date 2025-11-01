import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# load and clean data
df = pd.read_csv("Gathering Data/DraftStats.csv")
df.columns = df.columns.str.strip()

#convert to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

#drop non-numeric + invalid
df = df.dropna(subset=["Champion", "Role", "Winrate", "KDA"])

#keep only correlated columns
keep_cols = ["Champion", "Role", "KDA", "Wins", "GPM", "XPD@15", "DPM", "Winrate", "PrioScore"]
df = df[[c for c in keep_cols if c in df.columns]]
#group by role
roles = df["Role"].unique()
role_champs = {r: df[df["Role"] == r]["Champion"].tolist() for r in roles}

# generate synthetic matches
n_samples = 5000
teams = []
wins = []

for _ in range(n_samples):
    team = []
    for role in roles:
        champs = role_champs.get(role, [])
        if champs:
            team.append(random.choice(champs))
    if len(team) < len(roles):  # skip incomplete teams
        continue
    
    team_stats = df[df["Champion"].isin(team)]
    
    #compute average of selected features
    avg_stats = team_stats[["Winrate", "KDA", "GPM", "DPM", "XPD@15", "Wins", "PrioScore"]].mean()

    #create synthetic win probability
    score = (
        0.35 * (avg_stats["Winrate"] / 100)
        + 0.25 * (avg_stats["KDA"] / (avg_stats["KDA"] + 3))
        + 0.15 * (avg_stats["GPM"] / 600)
        + 0.15 * (avg_stats["DPM"] / 800)
        + 0.10 * (avg_stats["XPD@15"] / 1000)
    )
    win_prob = np.clip(score + np.random.normal(0, 0.05), 0, 1)
    
    teams.append(team)
    wins.append(1 if random.random() < win_prob else 0)

# one-hot encode team composition
all_champs = sorted(df["Champion"].unique())
data = []
for team, win in zip(teams, wins):
    row = {champ: 1 if champ in team else 0 for champ in all_champs}
    row["win"] = win
    data.append(row)

df_train = pd.DataFrame(data)

X = df_train.drop(columns=["win"])
y = df_train["win"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# random Forest with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\n✅ Best Parameters:", grid.best_params_)

# evaluate model
preds = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, preds))
print("AUC:", roc_auc_score(y_test, probs))
print("\nClassification Report:\n", classification_report(y_test, preds))

# predict a new team
example_team = ["Yunara", "Orianna", "Sejuani", "Aphelios", "Thresh"]
input_data = pd.DataFrame([[1 if champ in example_team else 0 for champ in X.columns]], columns=X.columns)
win_prob = best_model.predict_proba(input_data)[0, 1]
print(f"\nPredicted win probability for {example_team}: {win_prob:.3f}")
