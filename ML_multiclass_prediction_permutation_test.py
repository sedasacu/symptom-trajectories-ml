# =============================================================================
# Nested CV with permutation testing (macro F1, macro AUC & class-specific AUC)
# =============================================================================

# ---------------------------
# Imports
# ---------------------------
import numpy as np
import pandas as pd
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer

from tqdm import trange



# ---------------------------
# Load data
# ---------------------------
os.chdir('/Users/sedasacu/Desktop/Revision/analysis/data/features')
df = pd.read_excel('features_T1T6.xlsx')

X = df.drop(['ID', 'int_class', 'ext_class'], axis=1)
y = df['ext_class']   # or int_class


# ---------------------------
# Preprocessing
# ---------------------------
categorical_columns = []
for i in range(X.shape[1]):
    col = X.iloc[:, i]
    if col.min() == 0 and col.max() == 1:
        categorical_columns.append(i)

numerical_columns = [i for i in range(X.shape[1]) if i not in categorical_columns]

numerical_transformer = Pipeline([
    ('imputer', IterativeImputer(max_iter=50, random_state=42))
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

imputer = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# -----------------------------------------------------------
# Permutation test
# ============================================================
def permutation_test(
    pipeline,
    X,
    y,
    param_dist,
    n_permutations=100,
    random_state=42,
):

    rng = np.random.RandomState(random_state)

    perm_f1 = np.zeros(n_permutations)
    perm_auc = np.zeros(n_permutations)
    perm_class_auc = []  # <-- NEW

    for i in trange(n_permutations, desc="Permutation testing", ncols=100):
        y_perm = pd.Series(
            rng.permutation(y.values),
            index=y.index
        )

        f1, auc, class_auc = nested_cv_scalar(
            pipeline, X, y_perm, param_dist
        )

        perm_f1[i] = f1
        perm_auc[i] = auc
        perm_class_auc.append(class_auc)

    return perm_f1, perm_auc, np.array(perm_class_auc)



# ============================================================
# Nested CV returning scalar macro F1 & macro AUC
# ============================================================
def nested_cv_scalar(
    pipeline,
    X,
    y,
    param_dist,
    n_outer=10,
    n_inner=5,
    n_iter=50,
    random_state=42,
):

    outer_cv = StratifiedKFold(
        n_splits=n_outer, shuffle=True, random_state=random_state
    )
    inner_cv = StratifiedKFold(
        n_splits=n_inner, shuffle=True, random_state=random_state
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=inner_cv,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=random_state,
    )

    macro_f1_scores = []
    macro_auc_scores = []
    class_auc_scores = []  

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        search.fit(X_train, y_train)

        y_pred = search.predict(X_test)
        y_proba = search.predict_proba(X_test)

        macro_f1_scores.append(
            f1_score(y_test, y_pred, average='macro')
        )

        macro_auc_scores.append(
            roc_auc_score(
                y_test,
                y_proba,
                multi_class='ovr',
                average='macro'
            )
        )
        
    
        class_auc = roc_auc_score(
            y_test,
            y_proba,
            multi_class='ovr',
            average=None
        )
        class_auc_scores.append(class_auc)


    return np.mean(macro_f1_scores), np.mean(macro_auc_scores), np.mean(class_auc_scores, axis=0)


# ----------------------------------------------------
# INTERNALIZING SYMPTOMS TRAJECTORIES
# ----------------------------------------------------

# ---------------------------
# Model & hyperparameters
# ---------------------------
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

pipeline = Pipeline([
    ('imputer', imputer),
    ('scaler', MinMaxScaler()),
    ('model', model)
])

param_dist = {
    'model__solver': ['lbfgs', 'liblinear', 'newton-cg'],
    'model__class_weight': ['balanced', None],
    'model__C': np.logspace(-4, 4, 50)
}


# ----------------------------------------------------
# EXTERNALIZING SYMPTOMS TRAJECTORIES
# ----------------------------------------------------

# ---------------------------
# Model & hyperparameters
# ---------------------------
model=RandomForestClassifier(random_state=42)
pipeline= Pipeline([('imputer', imputer), ('scaler', MinMaxScaler()), ('model', model)])
# Note: Scaling is not necessary for RF, one might skip this step. 

# Define model space
from scipy.stats import randint as sp_randint
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 100)] # Number of trees in random forest
max_features = ['log2', 'sqrt'] # Number of features to consider at every split
max_depth = sp_randint (2,10) # Maximum number of levels in tree
min_samples_split = sp_randint(2,20) # Minimum number of samples required to split a node
min_samples_leaf = sp_randint(1,20) # Minimum number of samples required at each leaf node
class_weight=['balanced', None]

param_dist= {'model__n_estimators' : n_estimators,
             'model__max_features': max_features,
             'model__max_depth': max_depth,
             'model__min_samples_split': min_samples_split,
             'model__min_samples_leaf': min_samples_leaf,
             'model__class_weight': class_weight }



# ---------------------------
# Run permutation test
# ---------------------------


perm_f1, perm_auc, perm_class_auc = permutation_test(
    pipeline,
    X,
    y,
    param_dist,
    n_permutations=100   
)

print("\nPermutation test summary")
print("------------------------")
print(f"Macro F1  : mean = {perm_f1.mean():.3f}, SD = {perm_f1.std():.3f}")
print(f"Macro AUC : mean = {perm_auc.mean():.3f}, SD = {perm_auc.std():.3f}")

# Optional: 95% percentile interval
f1_low, f1_high = np.percentile(perm_f1, [2.5, 97.5])
auc_low, auc_high = np.percentile(perm_auc, [2.5, 97.5])

print("\n95% permutation intervals")
print("-------------------------")
print(f"Macro F1  : [{f1_low:.3f}, {f1_high:.3f}]")
print(f"Macro AUC : [{auc_low:.3f}, {auc_high:.3f}]")


class_labels = ['low', 'increasing', 'decreasing']
for i, cls in enumerate(class_labels):
    null_vals = perm_class_auc[:, i]

    mean_null = null_vals.mean()
    sd_null = null_vals.std()
    ci_low, ci_high = np.percentile(null_vals, [2.5, 97.5])

    print(
        f"{cls} null AUC: "
        f"mean={mean_null:.3f}, SD={sd_null:.3f}, "
        f"95% CI=[{ci_low:.3f}, {ci_high:.3f}]"
    )


# ============================================================
# Observed (true-label) performance
# ============================================================

observed_f1, observed_auc, observed_class_auc = nested_cv_scalar(
    pipeline, X, y, param_dist
)

print("\nObserved performance")
print("--------------------")
print(f"Macro F1  : {observed_f1:.3f}")
print(f"Macro AUC : {observed_auc:.3f}")



# ============================================================
# Empirical p-values
# ============================================================
p_f1 = (np.sum(perm_f1 >= observed_f1) + 1) / (len(perm_f1) + 1)
p_auc = (np.sum(perm_auc >= observed_auc) + 1) / (len(perm_auc) + 1)

print("\nPermutation test results")
print("------------------------")
print(f"p-value (Macro F1)  : {p_f1:.4f}")
print(f"p-value (Macro AUC) : {p_auc:.4f}")


print("\nNull distribution summary")
print("-------------------------")
print(f"Permuted Macro F1  : mean={perm_f1.mean():.3f}, std={perm_f1.std():.3f}")
print(f"Permuted Macro AUC : mean={perm_auc.mean():.3f}, std={perm_auc.std():.3f}")


print("\nClass-specific permutation test (AUC)")
print("------------------------------------")

for i, cls in enumerate(class_labels):
    p_val = (np.sum(perm_class_auc[:, i] >= observed_class_auc[i]) + 1) / (
        len(perm_class_auc) + 1
    )

    print(
        f"{cls}: Observed AUC = {observed_class_auc[i]:.3f}, "
        f"p-value = {p_val:.4f}"
    )

