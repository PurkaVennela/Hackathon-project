# Hackathon Project
#Python Script for detecting Multivariate Time Seies Anomaly Detection
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#File Paths---------------
input_csv = r"C:\Users\venne\Downloads\81ce1f00-c3f4-4baa-9b57-006fad1875adTEP_Train_Test.csv"
output_csv = r"C:\Users\venne\Downloads\TEP_Train_Test_Output.csv"

# -------------------------
# Step 1: To Load CSV
# -------------------------
df = pd.read_csv(input_csv)

# To Convert Time column to datetime and sort
df['Time'] = pd.to_datetime(df['Time'])
df = df.sort_values('Time').reset_index(drop=True)

# -------------------------
# Step 2: Handle missing/invalid values
# -------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# -------------------------
# Step 3: Split normal period for training
# -------------------------
train_df = df[(df['Time'] >= '2004-01-01 00:00:00') & (df['Time'] <= '2004-01-05 23:59:00')]
feature_cols = df.columns.drop('Time')
X_train = train_df[feature_cols].values

if X_train.shape[0] < 72:
    raise ValueError("Training period is less than 72 hours. Cannot train properly.")

# Remove constant features
non_constant_features = [col for col in feature_cols if df[col].std() > 0]
X_train = train_df[non_constant_features].values

# -------------------------
# Step 4: Scale features
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -------------------------
# Step 5: Train Isolation Forest
# -------------------------
iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
iso.fit(X_train_scaled)

# -------------------------
# Step 6: Predict anomaly scores for full dataset
# -------------------------
X_all_scaled = scaler.transform(df[non_constant_features].values)
raw_scores = -iso.decision_function(X_all_scaled)  # higher = more abnormal
percentile_scores = 100 * (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
percentile_scores += np.random.uniform(0, 0.01, size=percentile_scores.shape)  # tiny noise
df['abnormality_score'] = percentile_scores

# Training period validation
train_scores = df.loc[df['Time'].between('2004-01-01 00:00:00','2004-01-05 23:59:00'), 'abnormality_score']
if train_scores.mean() > 10 or train_scores.max() > 25:
    print("⚠ Warning: Training period scores exceed expected limits (mean <10, max <25)")

# -------------------------
# Step 7: Identify top 7 contributing features
# -------------------------
train_mean = X_train.mean(axis=0)
contribs = np.abs(df[non_constant_features].values - train_mean)

top_features_list = []
for row_idx in range(contribs.shape[0]):
    row_contrib = contribs[row_idx]
    total_contrib = row_contrib.sum()
    pct_contrib = (row_contrib / total_contrib) * 100 if total_contrib > 0 else np.zeros_like(row_contrib)
    
    valid_idx = np.where(pct_contrib > 1)[0]
    if len(valid_idx) > 0:
        valid_features = [(non_constant_features[i], pct_contrib[i]) for i in valid_idx]
        valid_features.sort(key=lambda x: (-x[1], x[0]))  # descending + alphabetical
        top_features = [f[0] for f in valid_features[:7]]
    else:
        top_features = []
    top_features += [''] * (7 - len(top_features))
    top_features_list.append(top_features)

for i in range(7):
    df[f'top_feature_{i+1}'] = [row[i] for row in top_features_list]

# -------------------------
# Step 8: Save final CSV
# -------------------------
df.to_csv(output_csv, index=False)
print(f"✅ Done! Output saved as '{output_csv}'")

# -------------------------
# Step 9: Visualizations
# -------------------------
# Scatter plot: anomaly scores over time
threshold = 10
colors = ['red' if s > threshold else 'green' for s in df['abnormality_score']]

plt.figure(figsize=(15,5))
plt.scatter(df['Time'], df['abnormality_score'], c=colors, s=10)
plt.axhline(y=threshold, color='blue', linestyle='--', label='Threshold')
plt.xlabel('Time')
plt.ylabel('Abnormality Score')
plt.title('Anomaly Scores Over Time (Red = Anomaly)')
plt.legend()
plt.show()

# Bar plot: top contributing features for the row with highest anomaly
row_idx = df['abnormality_score'].idxmax()
top_features = df.loc[row_idx, [f'top_feature_{i+1}' for i in range(7)]].values
top_features = [f for f in top_features if f != '']

if len(top_features) > 0:
    contrib_values = np.abs(df.loc[row_idx, top_features] - df[non_constant_features].mean()[top_features])
    plt.figure(figsize=(10,5))  # wider figure
plt.bar(top_features, contrib_values, color='orange')
plt.title(f'Top Contributing Features for Row {row_idx}')
plt.ylabel('Contribution (Deviation from Mean)')
plt.xticks(rotation=45, ha='right')  # rotate labels and align right
plt.tight_layout()  # adjust layout to prevent clipping
plt.show()
