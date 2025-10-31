import pandas as pd
import re

#read in the text file
with open("WorldsData.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

data = []

#define regex pattern to capture the data
pattern = re.compile(
    r"^([A-Za-z\.']+(?: [A-Za-z\.']+)*)\s+"    # Champion name (e.g. 'Jarvan IV')
    r"(\d+)\s+"                                # Picks
    r"(\d+)\s+"                                # Bans
    r"([\d\.]+%)\s+"                           # PrioScore
    r"(\d*)\s*"                                # Wins 
    r"(\d*)\s*"                                # Losses 
    r"([\d\.%]*)\s*"                           # Winrate
    r"([\d\.]*)\s*"                            # KDA
    r"([\d\.]*)\s*"                            # Average BT
    r"([\d\.]*)\s*"                            # Avg RP
    r"([\d:]+)\s*"                             # GT
    r"([\d\.]*)\s*"                            # CSM
    r"([\d\.]*)\s*"                            # DPM
    r"([\d\.]*)\s*"                            # GPM
    r"([\-]?\d+\.?\d*)?\s*"                    # CSD@15
    r"([\-]?\d+\.?\d*)?\s*"                    # GD@15
    r"([\-]?\d+\.?\d*)?"                       # XPD@15
)

#parse each line using the regex pattern
for line in lines:
    line = line.strip()
    if not line:
        continue
    match = pattern.match(line)
    if match:
        data.append(match.groups())

#create a DataFrame
columns = [
    "Champion", "Picks", "Bans", "PrioScore", "Wins", "Losses", "Winrate",
    "KDA", "Average BT", "Avg RP", "GT", "CSM", "DPM", "GPM",
    "CSD@15", "GD@15", "XPD@15"
]

df = pd.DataFrame(data, columns=columns)

#clean the data
df = df.fillna("")

#get rid of percentage signs and convert to appropriate types
df["PrioScore"] = df["PrioScore"].str.rstrip('%').astype(float)
df["Winrate"] = pd.to_numeric(df["Winrate"].str.replace("%", "", regex=False), errors="coerce")

#deduplicate repeated champ names
df["Champion"] = df["Champion"].apply(lambda x: " ".join(sorted(set(x.split()), key=x.split().index)))

#convert appropriate columns to numeric types
numeric_cols = [
   "Picks", "Bans", "PrioScore", "Wins", "Losses", "Winrate",
    "KDA", "Average BT", "Avg RP", "GT", "CSM", "DPM", "GPM",
    "CSD@15", "GD@15", "XPD@15"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

#conert to csv
df.to_csv("DraftStats.csv", index=False)
print("Data successfully written to DraftStats.csv")
print(df.dtypes)