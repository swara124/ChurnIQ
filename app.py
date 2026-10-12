from flask import Flask, render_template, request, redirect, session, send_file
from datetime import datetime
import mysql.connector
import pandas as pd
import pickle, os, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils import resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from churn_reasons import (get_reasons_for_customer, build_reason_data,
                            get_reason_summary, get_medians)

app = Flask(__name__)
app.secret_key = "secret123"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

# ── DB Connection with auto-reconnect ────────────────────────────────────────
def get_db():
    try:
        conn = mysql.connector.connect(
            host="localhost", user="root", password="root", database="churn_db"
        )
        return conn
    except Exception as e:
        print(f"DB connection error: {e}")
        return None

conn = get_db()
cursor = conn.cursor()

# ── Load model safely ────────────────────────────────────────────────────────
try:
    model, columns = pickle.load(open('model.pkl', 'rb'))
except Exception:
    model, columns = None, []

# ── Feature Engineering ──────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    if 'balance' in df.columns:
        if 'age' in df.columns:
            df['balance_per_age'] = df['balance'] / (df['age'] + 1)
        if 'tenure' in df.columns:
            df['balance_per_tenure'] = df['balance'] / (df['tenure'] + 1)
        df['high_balance']     = (df['balance'] > df['balance'].median()).astype(int)
        df['negative_balance'] = (df['balance'] < 0).astype(int)
    if 'tenure' in df.columns:
        df['is_new_customer']   = (df['tenure'] <= 1).astype(int)
        df['is_loyal_customer'] = (df['tenure'] >= 7).astype(int)
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0,30,45,60,100],
                                  labels=[0,1,2,3]).astype(float)
        df['senior'] = (df['age'] > 55).astype(int)
    if 'creditscore' in df.columns:
        df['poor_credit']      = (df['creditscore'] < 550).astype(int)
        df['excellent_credit'] = (df['creditscore'] > 750).astype(int)
    if 'numofproducts' in df.columns:
        df['multi_product'] = (df['numofproducts'] > 1).astype(int)
    if 'duration' in df.columns and 'campaign' in df.columns:
        df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
        df['high_engagement']       = (df['duration'] > df['duration'].median()).astype(int)
    if 'pdays' in df.columns:
        df['previously_contacted'] = (df['pdays'] != -1).astype(int)
    key = [c for c in ['balance','tenure','creditscore','age','duration'] if c in df.columns]
    for i in range(len(key)):
        for j in range(i+1, min(i+2, len(key))):
            df[f'{key[i]}_x_{key[j]}'] = df[key[i]] * df[key[j]]
    return df


def smart_encode(df):
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        n = df[col].nunique()
        if n == 2:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        elif n <= 10:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
        else:
            df = df.drop(col, axis=1)
    return df


def fill_missing(df):
    """
    FIX: Separate numeric and categorical null-filling.
    Numeric  → median
    Object/string → mode (most frequent value)
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            df[col] = df[col].fillna(df[col].median())
        else:
            # object / string / category columns
            mode_vals = df[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else 'unknown'
            df[col] = df[col].fillna(fill_val)
    return df


def plot_model_comparison(results_dict, filepath):
    models = list(results_dict.keys())
    f1s  = [v['f1']  for v in results_dict.values()]
    aucs = [v['auc'] for v in results_dict.values()]
    colors = ['#3b82f6','#8b5cf6','#06b6d4','#f59e0b']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f172a')
    for ax, scores, title in zip(axes, [f1s, aucs], ['F1 Score', 'ROC-AUC']):
        bars = ax.bar(models, scores, color=colors[:len(models)],
                      edgecolor='none', width=0.6)
        ax.set_facecolor('#1e293b')
        ax.set_title(title, color='white', fontsize=13, pad=12)
        ax.set_ylim(0, 1.1)
        ax.tick_params(colors='#94a3b8')
        ax.set_xticklabels(models, rotation=20, ha='right',
                           color='#94a3b8', fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
        ax.yaxis.set_tick_params(labelcolor='#94a3b8')
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f'{score:.2f}', ha='center', va='bottom',
                    color='white', fontsize=10, fontweight='bold')
    plt.suptitle('Model Performance Comparison', color='white', fontsize=15)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', facecolor='#0f172a', dpi=120)
    plt.close()


def plot_feature_importance(model, feature_names, filepath):
    if not hasattr(model, 'feature_importances_'):
        return
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-10:]
    top_f = [feature_names[i] for i in top_idx]
    top_s = importances[top_idx]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    bars = ax.barh(top_f, top_s, color='#3b82f6', edgecolor='none')
    ax.set_title('Top Feature Importances', color='white', fontsize=13, pad=10)
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    for bar, score in zip(bars, top_s):
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', color='white', fontsize=9)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', facecolor='#0f172a', dpi=120)
    plt.close()


def plot_reason_chart(reason_summary, filepath):
    if not reason_summary:
        return
    labels = [r[0] for r in reason_summary]
    counts = [r[1] for r in reason_summary]
    colors = ['#ef4444','#f59e0b','#8b5cf6','#3b82f6',
              '#06b6d4','#10b981','#f97316','#ec4899']
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    bars = ax.barh(labels[::-1], counts[::-1],
                   color=colors[:len(labels)][::-1],
                   edgecolor='none', height=0.6)
    ax.set_title('Top Churn Reasons (across at-risk customers)',
                 color='white', fontsize=13, pad=10)
    ax.tick_params(colors='#94a3b8', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.xaxis.set_tick_params(labelcolor='#94a3b8')
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.3,
                bar.get_y() + bar.get_height()/2,
                str(count), va='center', color='white',
                fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', facecolor='#0f172a', dpi=120)
    plt.close()


# ── Simple routes ────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (%s,%s,%s)",
            (request.form['name'], request.form['email'], request.form['password'])
        )
        conn.commit()
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['POST'])
def login():
    cursor.execute(
        "SELECT * FROM users WHERE email=%s AND password=%s",
        (request.form['email'], request.form['password'])
    )
    if cursor.fetchone():
        session['user'] = request.form['email']
        return redirect('/dashboard')
    return "Invalid Login"

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')
    return render_template('dashboard.html')

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect('/login')
    try:
        cursor.execute("SELECT id, name, email FROM users WHERE email=%s",
                       (session['user'],))
        u = cursor.fetchone()
        user_id, name, email = u
        cursor.execute("""
            SELECT u.file_name, u.upload_date,
                   p.total_customers, p.churn_count, p.non_churn_count
            FROM uploads u
            LEFT JOIN predictions p ON u.id = p.upload_id
            WHERE u.user_id=%s
            ORDER BY u.upload_date DESC
        """, (user_id,))
        raw = cursor.fetchall()
        history = [(r[0], r[1], r[2] or 0, r[3] or 0, r[4] or 0) for r in raw]
        return render_template('profile.html', name=name,
                               email=email, history=history)
    except Exception as e:
        return f"Error: {e}"

@app.route('/result')
def result():
    if 'result' not in session:
        return redirect('/upload')
    return render_template('result.html', **session['result'])

@app.route('/upload', methods=['GET'])
def upload_page():
    if 'user' not in session:
        return redirect('/login')
    return render_template('upload.html')

@app.route('/download')
def download():
    return send_file("churn.csv", as_attachment=True)

@app.route('/download_model_graph')
def download_model_graph():
    return send_file("static/model_comparison.png", as_attachment=True)

@app.route('/download_feature_graph')
def download_feature_graph():
    return send_file("static/feature_importance.png", as_attachment=True)

@app.route('/download_reason_graph')
def download_reason_graph():
    return send_file("static/reason_chart.png", as_attachment=True)


# ── Main upload & predict route ──────────────────────────────────────────────
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'user' not in session:
        return redirect('/login')
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return "No file uploaded"

        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        original_df = df.copy()

        # Normalize column names
        df.columns = (df.columns.str.strip()
                                .str.lower()
                                .str.replace(' ', '_')
                                .str.replace('-', '_'))
        original_df.columns = df.columns.copy()

        # Detect ID column
        id_column = None
        for col in df.columns:
            if any(x in col for x in
                   ['customerid', 'customer_id', 'rownum', 'id', 'name', 'customer']):
                id_column = col
                break

        # DB: get user id & log upload
        email = session['user']
        cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
        user_id = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO uploads (user_id, file_name, file_path, upload_date) "
            "VALUES (%s,%s,%s,%s)",
            (user_id, filename, filepath, datetime.now())
        )
        conn.commit()
        upload_id = cursor.lastrowid

        # ── FIX: Smart null filling (numeric=median, text=mode) ─────────────
        df = fill_missing(df)

        # Drop target columns if present in uploaded data
        drop_target = [c for c in df.columns
                       if c in ['exited', 'churn', 'churned', 'term_deposit', 'y']]
        df_raw_for_reasons = df.drop(
            columns=drop_target + ([id_column] if id_column else []),
            errors='ignore'
        ).copy()

        # Compute medians on raw numeric cols for reason rules
        raw_medians = get_medians(df_raw_for_reasons)

        # Feature engineering & encoding
        df_eng  = engineer_features(df_raw_for_reasons.copy())
        df_enc  = smart_encode(df_eng)
        df_model = df_enc.select_dtypes(include=[np.number]).copy()
        df_model = df_model.loc[:, df_model.nunique() > 1]

        # ── KMeans clustering to generate churn labels ───────────────────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_model)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        df_model['cluster'] = cluster_labels

        # Cluster with lower mean = churn cluster
        cmeans = {
            cl: df_model[df_model['cluster'] == cl]
                .drop('cluster', axis=1).mean().mean()
            for cl in [0, 1]
        }
        churn_cluster = min(cmeans, key=cmeans.get)
        df_model['churn'] = (df_model['cluster'] == churn_cluster).astype(int)

        # Add small noise to avoid perfect separation
        noise = np.random.rand(len(df_model)) < 0.08
        df_model.loc[noise, 'churn'] = 1 - df_model.loc[noise, 'churn']

        # Feature selection
        X_raw2 = df_model.drop(['churn', 'cluster'], axis=1)
        y = df_model['churn']
        k = min(15, X_raw2.shape[1])
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_sel = selector.fit_transform(X_raw2, y)
        sel_feats = X_raw2.columns[selector.get_support()].tolist()
        X = pd.DataFrame(X_sel, columns=sel_feats)

        # Train / test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Balance classes
        tdf = pd.concat([X_train, y_train.rename('churn')], axis=1)
        maj  = tdf[tdf.churn == 0]
        minn = tdf[tdf.churn == 1]
        if len(maj) > len(minn):
            maj = resample(maj, replace=False,
                           n_samples=len(minn), random_state=42)
        else:
            minn = resample(minn, replace=True,
                            n_samples=len(maj), random_state=42)
        bal = pd.concat([maj, minn]).sample(frac=1, random_state=42)
        X_train, y_train = bal.drop('churn', axis=1), bal['churn']

        # ── Train models ────────────────────────────────────────────────────
        models_dict = {
            'Logistic Reg':   LogisticRegression(max_iter=500, C=0.1),
            'Random Forest':  RandomForestClassifier(n_estimators=100,
                                                     max_depth=5,
                                                     random_state=42),
            'Gradient Boost': GradientBoostingClassifier(n_estimators=100,
                                                         max_depth=3,
                                                         learning_rate=0.1,
                                                         random_state=42),
        }
        model_results = {}
        trained_models = {}
        for name, m in models_dict.items():
            m.fit(X_train, y_train)
            yp  = m.predict(X_test)
            ypr = (m.predict_proba(X_test)[:, 1]
                   if hasattr(m, 'predict_proba') else yp.astype(float))
            model_results[name] = dict(
                f1=round(f1_score(y_test, yp), 4),
                auc=round(roc_auc_score(y_test, ypr), 4),
                acc=round(accuracy_score(y_test, yp), 4),
                precision=round(precision_score(y_test, yp, zero_division=0), 4),
                recall=round(recall_score(y_test, yp, zero_division=0), 4)
            )
            trained_models[name] = m

        # Ensemble
        ensemble = VotingClassifier(
            estimators=list(trained_models.items()), voting='soft'
        )
        ensemble.fit(X_train, y_train)
        yp_e  = ensemble.predict(X_test)
        ypr_e = ensemble.predict_proba(X_test)[:, 1]
        model_results['Ensemble'] = dict(
            f1=round(f1_score(y_test, yp_e), 4),
            auc=round(roc_auc_score(y_test, ypr_e), 4),
            acc=round(accuracy_score(y_test, yp_e), 4),
            precision=round(precision_score(y_test, yp_e, zero_division=0), 4),
            recall=round(recall_score(y_test, yp_e, zero_division=0), 4)
        )

        # Pick best model
        best_name  = max(model_results, key=lambda k: model_results[k]['f1'])
        final_model = (ensemble if best_name == 'Ensemble'
                       else trained_models[best_name])
        best = model_results[best_name]
        cm = confusion_matrix(y_test, final_model.predict(X_test))
        tn, fp, fn, tp_val = cm.ravel()

        # Feature importance for reason engine
        rf_model = trained_models['Random Forest']
        feat_imp_dict = dict(zip(sel_feats, rf_model.feature_importances_))
        for raw_col in df_raw_for_reasons.select_dtypes(include=[np.number]).columns:
            if raw_col not in feat_imp_dict:
                feat_imp_dict[raw_col] = 0.01

        # ── Predictions on full dataset ──────────────────────────────────────
        predictions  = final_model.predict(X)
        churn_probs  = final_model.predict_proba(X)[:, 1]
        df['prediction']    = predictions
        df['churn_risk_pct'] = (churn_probs * 100).round(1)
        if id_column:
            df[id_column] = original_df[id_column]

        churn_df   = df[df['prediction'] == 1].sort_values(
            'churn_risk_pct', ascending=False
        )
        total      = len(df)
        churn_cnt  = len(churn_df)
        non_churn  = total - churn_cnt
        churn_rate = round(churn_cnt / total * 100, 1) if total else 0

        # ── Reason generation ────────────────────────────────────────────────
        churn_raw_df = df_raw_for_reasons.iloc[churn_df.index].copy()
        churn_raw_df['churn_risk_pct'] = churn_df['churn_risk_pct'].values

        all_reasons_raw = []
        for _, row in churn_raw_df.iterrows():
            reasons = get_reasons_for_customer(row, feat_imp_dict, top_n=3)
            all_reasons_raw.append(reasons)

        reason_summary = get_reason_summary(all_reasons_raw)

        # Build churn list
        ids = (churn_df[id_column].astype(str).tolist()
               if id_column else churn_df.index.astype(str).tolist())
        risks = churn_df['churn_risk_pct'].tolist()
        churn_list = []
        for i, (cid, risk) in enumerate(zip(ids, risks)):
            reasons = all_reasons_raw[i] if i < len(all_reasons_raw) else []
            churn_list.append({'id': cid, 'risk': risk, 'reasons': reasons})

        # ── Save charts ──────────────────────────────────────────────────────
        plot_model_comparison(model_results, 'static/model_comparison.png')
        plot_feature_importance(rf_model, sel_feats, 'static/feature_importance.png')
        plot_reason_chart(reason_summary, 'static/reason_chart.png')

        # Save churn CSV
        churn_df.to_csv("churn.csv", index=False)

        # Store in DB
        cursor.execute(
            "INSERT INTO predictions "
            "(user_id, upload_id, total_customers, churn_count, "
            " non_churn_count, accuracy) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (user_id, upload_id, total, churn_cnt,
             non_churn, float(best['acc']))
        )
        conn.commit()

        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

        return render_template(
            'result.html',
            total=total, churn=churn_cnt, non_churn=non_churn,
            churn_rate=churn_rate,
            accuracy=round(best['acc']   * 100, 2),
            f1=round(best['f1']          * 100, 2),
            auc=round(best['auc']        * 100, 2),
            precision=round(best['precision'] * 100, 2),
            recall=round(best['recall']  * 100, 2),
            churn_list=churn_list[:60],
            id_column=id_column,
            tp=int(tp_val), tn=int(tn), fp=int(fp), fn=int(fn),
            best_model=best_name,
            model_results=model_results,
            n_features=len(sel_feats),
            top_features=sel_feats[:5],
            reason_summary=reason_summary
        )

    except Exception as e:
        import traceback
        return f"<pre>Error: {e}\n{traceback.format_exc()}</pre>"


# ── Report route ─────────────────────────────────────────────────────────────
@app.route('/report')
def report():
    if 'user' not in session:
        return redirect('/login')
    cursor.execute("SELECT id FROM users WHERE email=%s", (session['user'],))
    user_id = cursor.fetchone()[0]
    cursor.execute("""
        SELECT u.file_name, u.upload_date,
               p.total_customers, p.churn_count, p.non_churn_count
        FROM uploads u
        JOIN predictions p ON u.id = p.upload_id
        WHERE u.user_id=%s
        ORDER BY u.upload_date ASC
    """, (user_id,))
    history = cursor.fetchall()
    total_churn     = sum(r[3] or 0 for r in history)
    total_customers = sum(r[2] or 0 for r in history)
    return render_template(
        'report.html',
        total_uploads=len(history),
        total_customers=total_customers,
        total_churn=total_churn,
        total_non_churn=sum(r[4] or 0 for r in history),
        dates=[r[1].strftime('%d-%m-%Y') for r in history if r[1]],
        churn_values=[r[3] or 0 for r in history],
        non_churn_values=[r[4] or 0 for r in history],
        file_names=[r[0] for r in history]
    )


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


if __name__ == '__main__':
    app.run(debug=True)
