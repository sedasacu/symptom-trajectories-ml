# Project: Predicting psychopathology symptom trajectories using ML models
# published in Journal Name (Year)
# Script written by Seda Sacu
# Contact: sedasacu@gmail.com 


# ------------------------------------------------
# Load data
# ------------------------------------------------

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler #scaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer


# ------------------------------------------------
# Load data
# ------------------------------------------------
## Change directory
import os
os.chdir('/Users/sedasacu/Desktop/Revision/analysis/data/Theoretical') 

## Import data
df = pd.read_excel('features_T1T6.xlsx')

## Data check 
# See first 5 rows
df.head()
# See missing values
df.isnull().sum()

# Declare feature vector and target variable
X = df.drop(['ID','int_class','ext_class'], axis=1)
y = df['int_class'] # can be changed to ext_class


# ------------------------------------------------
# Preprocessing
# ------------------------------------------------

## Define the imputation strategy

categorical_columns = []

# Iterate through the columns of the DataFrame
for i in range(X.shape[1]):  # Loop over the number of columns
    col = X.iloc[:, i]  # Select the i-th column
    if col.min() == 0 and col.max() == 1:  # Check if min is 0 and max is 1
        categorical_columns.append(i)  # Append the column index to the list


numerical_columns = [i for i in range(len(X.columns)) if i not in categorical_columns]

# Define the preprocessing steps for numerical and categorical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer(max_iter=50, random_state=42)),
    ])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
   ])

# Define the column transformer to apply different preprocessing steps to numerical and categorical columns
imputer = ColumnTransformer(
      transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
        ])


# ------------------------------------------------
# Function to calculate bootstrapped CI
# ------------------------------------------------

def bootstrap_ci(values, n_bootstrap=5000, ci=0.95, random_state=42):
    """
    Percentile bootstrap confidence interval for the mean.
    Returns: mean, SD, CI lower, CI upper
    """
    rng = np.random.RandomState(random_state)
    values = np.asarray(values)

    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])

    lower = np.percentile(boot_means, 100 * (1 - ci) / 2)
    upper = np.percentile(boot_means, 100 * (1 + ci) / 2)

    return np.mean(values), np.std(values, ddof=1), lower, upper


# ------------------------------------------------
# Nested Cross Validation
# ------------------------------------------------

def nested_cv(
    model,
    X,
    y,
    param_dist,
    n_outer=10,
    n_inner=5,
    n_iter=50,
    random_state=42,
):

    cv_outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    cv_inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=random_state)


    # Initialize random search
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_inner,
        n_jobs=-1,
        scoring='f1_macro',
        random_state=random_state,
    )

    micro_roc_auc=[]
    macro_roc_auc=[]
    roc_auc_class=[]
    f1_weighted=[]
    f1_macro=[]
    f1_scores=[]

    for train_index, test_index in cv_outer.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    

        # fit the model
        search.fit(X_train, y_train)


        # Evaluate model
        y_pred = search.predict(X_test)
        
        # Printing fold-specific metrics
        fold_idx = len(f1_scores) + 1
        print(f"These are the results for fold {fold_idx}")
        
        # Calculate and print class-specific f1 score
        f1_score_class=f1_score(y_test,y_pred, average=None)
        f1_scores.append(f1_score_class)
        print("F1_scores:", f1_score_class)


        # Calculate and print f1 Weighted
        f1_weighted_score=f1_score(y_test, y_pred, average='weighted')
        f1_weighted.append(f1_weighted_score)
        print("F1_weighted:", f1_weighted_score)

        # Calculate and print f1 Macro
        f1_macro_score=f1_score(y_test, y_pred, average='macro')
        f1_macro.append(f1_macro_score)
        print("F1_macro:", f1_macro_score)

       
        # Calculate ROC AUC score
        y_test_pred_proba=search.predict_proba(X_test)
        
        # Class-specific ROC AUC
        roc_auc_score_class=roc_auc_score(y_test, y_test_pred_proba, multi_class="ovr", average=None)
        #print("ROC_AUC_scores:", roc_auc_score_class)


        # micro roc auc

        micro_roc_auc_ovr = roc_auc_score(
            y_test,
            y_test_pred_proba,
            multi_class="ovr",
            average="micro",
        )

        # macro roc auc

        macro_roc_auc_ovr = roc_auc_score(
            y_test,
            y_test_pred_proba,
            multi_class="ovr",
            average="macro",
        )

        roc_auc_class.append(roc_auc_score_class)
        micro_roc_auc.append(micro_roc_auc_ovr)
        macro_roc_auc.append(macro_roc_auc_ovr)

    
    return pd.DataFrame(f1_scores), pd.DataFrame(roc_auc_class), micro_roc_auc, macro_roc_auc, f1_weighted, f1_macro




# ------------------------------------------------
# Model + hyperparameters
# ------------------------------------------------

############### LOGISTIC REGRESSION ##############

model=LogisticRegression(max_iter=1000, random_state=42)
pipeline= Pipeline([('imputer', imputer), ('scaler', MinMaxScaler()), ('model', model)])

# Define model space
solver=['lbfgs','liblinear', 'newton-cg'] 
class_weight=['balanced', None]
C= np.logspace(-4,4,50)  

param_dist= {'model__solver' : solver,
             'model__class_weight': class_weight,
             'model__C' : C }


# Run CV
f1_scores_LR, roc_auc_class_LR, micro_roc_auc_LR, macro_roc_auc_LR, f1_weighted_LR, f1_macro_LR = nested_cv(pipeline, X, y, param_dist)


# Model Evaluation
roc_auc_arr = np.array(roc_auc_class_LR)  # shape: (n_folds, n_classes)
class_labels = ['low', 'increasing', 'decreasing']

# Class-specific ROC AUC 
class_auc_results = []

for i, cls in enumerate(class_labels):
    mean_auc, sd_auc, ci_low, ci_high = bootstrap_ci(
        roc_auc_arr[:, i],
        n_bootstrap=5000
    )

    class_auc_results.append({
        'class': cls,
        'AUC_mean': mean_auc,
        'AUC_SD': sd_auc,
        'AUC_95CI_low': ci_low,
        'AUC_95CI_high': ci_high
    })

class_auc_df = pd.DataFrame(class_auc_results)
print(class_auc_df)

# Global ROC AUC
global_auc_results = []

for name, values in zip(
    ['ROC_AUC_micro', 'ROC_AUC_macro'],
    [micro_roc_auc_LR, macro_roc_auc_LR]
):
    mean_auc, sd_auc, ci_low, ci_high = bootstrap_ci(
        values,
        n_bootstrap=5000
    )

    global_auc_results.append({
        'metric': name,
        'mean': mean_auc,
        'SD': sd_auc,
        '95CI_low': ci_low,
        '95CI_high': ci_high
    })

global_auc_df = pd.DataFrame(global_auc_results)
print(global_auc_df)

# Class-specific f1 scores
f1_arr = np.array(f1_scores_LR)
class_labels = ['low', 'increasing', 'decreasing']

class_f1_results = []

for i, cls in enumerate(class_labels):
    mean_f1, sd_f1, ci_low, ci_high = bootstrap_ci(
        f1_arr[:, i],
        n_bootstrap=5000
    )
    class_f1_results.append({
        'class': cls,
        'f1_mean': mean_f1,
        'f1_SD': sd_f1,
        'f1_95CI_low': ci_low,
        'f1_95CI_high': ci_high
    })

class_f1_df = pd.DataFrame(class_f1_results)
print(class_f1_df)


# Global f1 scores
global_f1_results = []

for name, values in zip(
    ['f1_weighted', 'f1_macro'],
    [f1_weighted_LR, f1_macro_LR]
):
    mean_f1, sd_f1, ci_low, ci_high = bootstrap_ci(
        values,
        n_bootstrap=5000
    )

    global_f1_results.append({
        'metric': name,
        'mean': mean_f1,
        'SD': sd_f1,
        '95CI_low': ci_low,
        '95CI_high': ci_high
    })

global_f1_df = pd.DataFrame(global_f1_results)
print(global_f1_df)




################ SUPPORT VECTOR CLASSIFIER  ##################

model=SVC(random_state=42, probability=True)
pipeline= Pipeline([('imputer', imputer), ('scaler', MinMaxScaler()), ('model', model)])

# Define model space
from scipy.stats import uniform
kernel=['linear', 'rbf', 'poly']
C= np.logspace (-4,4,50)
gamma=['scale', 'auto'] + list(np.logspace(-3, 3, 50))
class_weight=['balanced', None]
decision_function_shape= ['ovo','ovr']

param_dist= {'model__kernel' : kernel,
             'model__C': C,
             'model__gamma': gamma,
             'model__class_weight': class_weight,
             'model__decision_function_shape': decision_function_shape}


# Run CV
f1_scores_SVC, roc_auc_class_SVC, micro_roc_auc_SVC, macro_roc_auc_SVC, f1_weighted_SVC, f1_macro_SVC = nested_cv(pipeline, X, y, param_dist)

# Model Evaluation
roc_auc_arr = np.array(roc_auc_class_SVC)  # shape: (n_folds, n_classes)
class_labels = ['low', 'increasing', 'decreasing']

# Class-specific ROC AUC scores
class_auc_results = []

for i, cls in enumerate(class_labels):
    mean_auc, sd_auc, ci_low, ci_high = bootstrap_ci(
        roc_auc_arr[:, i],
        n_bootstrap=5000
    )

    class_auc_results.append({
        'class': cls,
        'AUC_mean': mean_auc,
        'AUC_SD': sd_auc,
        'AUC_95CI_low': ci_low,
        'AUC_95CI_high': ci_high
    })

class_auc_df = pd.DataFrame(class_auc_results)
print(class_auc_df)

# Global ROC AUC scores
global_auc_results = []

for name, values in zip(
    ['ROC_AUC_micro', 'ROC_AUC_macro'],
    [micro_roc_auc_SVC, macro_roc_auc_SVC]
):
    mean_auc, sd_auc, ci_low, ci_high = bootstrap_ci(
        values,
        n_bootstrap=5000
    )

    global_auc_results.append({
        'metric': name,
        'mean': mean_auc,
        'SD': sd_auc,
        '95CI_low': ci_low,
        '95CI_high': ci_high
    })

global_auc_df = pd.DataFrame(global_auc_results)
print(global_auc_df)

# Class-specific f1 scores
f1_arr = np.array(f1_scores_SVC)
class_labels = ['low', 'increasing', 'decreasing']

class_f1_results = []

for i, cls in enumerate(class_labels):
    mean_f1, sd_f1, ci_low, ci_high = bootstrap_ci(
        f1_arr[:, i],
        n_bootstrap=5000
    )
    class_f1_results.append({
        'class': cls,
        'f1_mean': mean_f1,
        'f1_SD': sd_f1,
        'f1_95CI_low': ci_low,
        'f1_95CI_high': ci_high
    })

class_f1_df = pd.DataFrame(class_f1_results)
print(class_f1_df)

# Global f1 scores
global_f1_results = []

for name, values in zip(
    ['f1_weighted', 'f1_macro'],
    [f1_weighted_SVC, f1_macro_SVC]
):
    mean_f1, sd_f1, ci_low, ci_high = bootstrap_ci(
        values,
        n_bootstrap=5000
    )

    global_f1_results.append({
        'metric': name,
        'mean': mean_f1,
        'SD': sd_f1,
        '95CI_low': ci_low,
        '95CI_high': ci_high
    })

global_f1_df = pd.DataFrame(global_f1_results)
print(global_f1_df)



#################### RANDOM FOREST  ######################

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

# Run CV
f1_scores_RF, roc_auc_class_RF, micro_roc_auc_RF, macro_roc_auc_RF, f1_weighted_RF, f1_macro_RF = nested_cv(pipeline, X, y, param_dist)


# Model Evaluation
roc_auc_arr = np.array(roc_auc_class_RF)  # shape: (n_folds, n_classes)
class_labels = ['low', 'increasing', 'decreasing']

# Class-specific ROC AUC scores
class_auc_results = []

for i, cls in enumerate(class_labels):
    mean_auc, sd_auc, ci_low, ci_high = bootstrap_ci(
        roc_auc_arr[:, i],
        n_bootstrap=5000
    )

    class_auc_results.append({
        'class': cls,
        'AUC_mean': mean_auc,
        'AUC_SD': sd_auc,
        'AUC_95CI_low': ci_low,
        'AUC_95CI_high': ci_high
    })

class_auc_df = pd.DataFrame(class_auc_results)
print(class_auc_df)

# Global ROC AUC scores
global_auc_results = []

for name, values in zip(
    ['ROC_AUC_micro', 'ROC_AUC_macro'],
    [micro_roc_auc_RF, macro_roc_auc_RF]
):
    mean_auc, sd_auc, ci_low, ci_high = bootstrap_ci(
        values,
        n_bootstrap=5000
    )

    global_auc_results.append({
        'metric': name,
        'mean': mean_auc,
        'SD': sd_auc,
        '95CI_low': ci_low,
        '95CI_high': ci_high
    })

global_auc_df = pd.DataFrame(global_auc_results)
print(global_auc_df)

# Class-specific f1 scores
f1_arr = np.array(f1_scores_RF)
class_labels = ['low', 'increasing', 'decreasing']

class_f1_results = []

for i, cls in enumerate(class_labels):
    mean_f1, sd_f1, ci_low, ci_high = bootstrap_ci(
        f1_arr[:, i],
        n_bootstrap=5000
    )
    class_f1_results.append({
        'class': cls,
        'f1_mean': mean_f1,
        'f1_SD': sd_f1,
        'f1_95CI_low': ci_low,
        'f1_95CI_high': ci_high
    })

class_f1_df = pd.DataFrame(class_f1_results)
print(class_f1_df)


# Global f1 scores
global_f1_results = []

for name, values in zip(
    ['f1_weighted', 'f1_macro'],
    [f1_weighted_RF, f1_macro_RF]
):
    mean_f1, sd_f1, ci_low, ci_high = bootstrap_ci(
        values,
        n_bootstrap=5000
    )

    global_f1_results.append({
        'metric': name,
        'mean': mean_f1,
        'SD': sd_f1,
        '95CI_low': ci_low,
        '95CI_high': ci_high
    })

global_f1_df = pd.DataFrame(global_f1_results)
print(global_f1_df)




################ eXtreme Gradient Boosting (XGBoost) ###############
model=XGBClassifier(random_state=42, eval_metric='mlogloss')
pipeline= Pipeline([('imputer', imputer), ('scaler', MinMaxScaler()), ('model', model)])

## XGBoost for RandomizedSearchCV
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

# Define model space
learning_rate = sp_randFloat(0.01,0.9)
max_depth =sp_randInt(3,10)
min_child_weight= sp_randInt(1,10)
subsample=np.arange(0.6, 1, 0.1)
objective=['multi:softmax', 'multi:softprob']

param_dist= {'model__learning_rate' : learning_rate,
             'model__max_depth': max_depth,
             'model__min_child_weight': min_child_weight,
             'model__subsample': subsample,
             'model__objective': objective,
             }

# Run CV
f1_scores_xgb, roc_auc_class_xgb, micro_roc_auc_xgb, macro_roc_auc_xgb, f1_weighted_xgb, f1_macro_xgb = nested_cv(pipeline, X, y, param_dist)

# Model Evaluation
roc_auc_arr = np.array(roc_auc_class_xgb)  # shape: (n_folds, n_classes)
class_labels = ['low', 'increasing', 'decreasing']

# Class-specific ROC AUC scores
class_auc_results = []

for i, cls in enumerate(class_labels):
    mean_auc, sd_auc, ci_low, ci_high = bootstrap_ci(
        roc_auc_arr[:, i],
        n_bootstrap=5000
    )

    class_auc_results.append({
        'class': cls,
        'AUC_mean': mean_auc,
        'AUC_SD': sd_auc,
        'AUC_95CI_low': ci_low,
        'AUC_95CI_high': ci_high
    })

class_auc_df = pd.DataFrame(class_auc_results)
print(class_auc_df)

# Global ROC AUC scores
global_auc_results = []

for name, values in zip(
    ['ROC_AUC_micro', 'ROC_AUC_macro'],
    [micro_roc_auc_xgb, macro_roc_auc_xgb]
):
    mean_auc, sd_auc, ci_low, ci_high = bootstrap_ci(
        values,
        n_bootstrap=5000
    )

    global_auc_results.append({
        'metric': name,
        'mean': mean_auc,
        'SD': sd_auc,
        '95CI_low': ci_low,
        '95CI_high': ci_high
    })

global_auc_df = pd.DataFrame(global_auc_results)
print(global_auc_df)

# Class-specific f1 scores
f1_arr = np.array(f1_scores_xgb)
class_labels = ['low', 'increasing', 'decreasing']

class_f1_results = []

for i, cls in enumerate(class_labels):
    mean_f1, sd_f1, ci_low, ci_high = bootstrap_ci(
        f1_arr[:, i],
        n_bootstrap=5000
    )
    class_f1_results.append({
        'class': cls,
        'f1_mean': mean_f1,
        'f1_SD': sd_f1,
        'f1_95CI_low': ci_low,
        'f1_95CI_high': ci_high
    })

class_f1_df = pd.DataFrame(class_f1_results)
print(class_f1_df)


# Global f1 scores
global_f1_results = []

for name, values in zip(
    ['f1_weighted', 'f1_macro'],
    [f1_weighted_xgb, f1_macro_xgb]
):
    mean_f1, sd_f1, ci_low, ci_high = bootstrap_ci(
        values,
        n_bootstrap=5000
    )

    global_f1_results.append({
        'metric': name,
        'mean': mean_f1,
        'SD': sd_f1,
        '95CI_low': ci_low,
        '95CI_high': ci_high
    })

global_f1_df = pd.DataFrame(global_f1_results)
print(global_f1_df)