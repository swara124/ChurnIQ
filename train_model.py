"""
Enhanced train_model.py — builds model.pkl with feature engineering + mutual info selection
"""
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('uploads/BankCustomerData.csv')
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

# Drop IDs
drop_cols = [c for c in data.columns if any(x in c for x in ['id', 'name', 'customer'])]
drop_cols += [c for c in data.columns if c in ['exited','churn','churned','term_deposit','y']]
data = data.drop(columns=drop_cols, errors='ignore')

# Fill missing
for col in data.columns:
    if data[col].dtype == object:
        data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'unknown')
    else:
        data[col] = data[col].fillna(data[col].median())

# Feature engineering
if 'balance' in data.columns and 'age' in data.columns:
    data['balance_per_age'] = data['balance'] / (data['age'] + 1)
if 'balance' in data.columns and 'tenure' in data.columns:
    data['balance_per_tenure'] = data['balance'] / (data['tenure'] + 1)
if 'balance' in data.columns:
    data['high_balance'] = (data['balance'] > data['balance'].median()).astype(int)
if 'tenure' in data.columns:
    data['is_loyal'] = (data['tenure'] >= 7).astype(int)
if 'age' in data.columns:
    data['senior'] = (data['age'] > 55).astype(int)

# Encode
for col in data.select_dtypes(include=['object']).columns:
    n = data[col].nunique()
    if n == 2:
        le = LabelEncoder(); data[col] = le.fit_transform(data[col].astype(str))
    elif n <= 10:
        dummies = pd.get_dummies(data[col].astype(str), prefix=col, drop_first=True)
        data = pd.concat([data.drop(col, axis=1), dummies], axis=1)
    else:
        data = data.drop(col, axis=1)

data = data.select_dtypes(include=[np.number])
data = data.loc[:, data.nunique() > 1]

# Cluster for labels
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
data['churn'] = kmeans.fit_predict(X_scaled)
cluster_means = {cl: data[data['churn']==cl].drop('churn',axis=1).mean().mean() for cl in [0,1]}
churn_cluster = min(cluster_means, key=cluster_means.get)
data['churn'] = (data['churn'] == churn_cluster).astype(int)

X = data.drop('churn', axis=1); y = data['churn']
selector = SelectKBest(score_func=mutual_info_classif, k=min(15, X.shape[1]))
selector.fit(X, y)
selected_cols = X.columns[selector.get_support()].tolist()
X_sel = X[selected_cols]

X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.25, random_state=42, stratify=y)
train_df = pd.concat([X_train, y_train.rename('churn')], axis=1)
maj = train_df[train_df.churn==0]; minn = train_df[train_df.churn==1]
if len(maj)>len(minn): maj = resample(maj, replace=False, n_samples=len(minn), random_state=42)
else: minn = resample(minn, replace=True, n_samples=len(maj), random_state=42)
balanced = pd.concat([maj,minn])
X_train = balanced.drop('churn',axis=1); y_train = balanced['churn']

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
lr = LogisticRegression(max_iter=500, C=0.1)
ensemble = VotingClassifier(estimators=[('rf',rf),('gb',gb),('lr',lr)], voting='soft')
ensemble.fit(X_train, y_train)

pickle.dump((ensemble, selected_cols), open('model.pkl', 'wb'))
print("✅ Enhanced model trained! Features:", selected_cols)
