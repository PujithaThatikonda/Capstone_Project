import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib


# ============================================================
# ZEPTO MODULE 2 - PART B
# PREDICTIVE MODELING
# ============================================================

print("=" * 80)
print("ZEPTO MODULE 2 - PART B: PREDICTIVE MODELING")
print("=" * 80)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

cleaned_file = BASE_DIR / "cleaned_titanic.csv"

if not cleaned_file.exists():

    print("\nERROR: cleaned_titanic.csv was not found.")

    print("Run 01_eda.py first.")

    raise SystemExit


# ============================================================
# LOAD THE CLEANED DATASET
# ============================================================

df = pd.read_csv(cleaned_file)

print("\nCleaned Titanic dataset loaded.")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_dir = BASE_DIR / "model_outputs"

output_dir.mkdir(exist_ok=True)


# ============================================================
# TASK 7
# STRATIFIED TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("TASK 7 - STRATIFIED TRAIN / TEST SPLIT")
print("=" * 80)


target = "survived"


# Classification features
classification_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]


X = df[classification_features]

y = df[target]


print("\nClass balance before split:")

class_balance = (
    y.value_counts(normalize=True)
    .sort_index()
    * 100
)

print(class_balance.round(2))

print("""
Stratification is important because the Titanic dataset is
imbalanced: approximately 62% of passengers did not survive
and 38% survived. Using stratify=y keeps approximately the
same class proportions in both the training and testing sets.
""")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))

print("\nTraining class balance:")
print(
    y_train.value_counts(normalize=True)
    .sort_index()
    .round(4)
)

print("\nTesting class balance:")
print(
    y_test.value_counts(normalize=True)
    .sort_index()
    .round(4)
)


# ============================================================
# TASK 8
# PREPROCESSING
# ============================================================

print("\n" + "=" * 80)
print("TASK 8 - TRAINING-ONLY PREPROCESSING")
print("=" * 80)


numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]


categorical_features = [
    "sex",
    "embarked"
]


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


print("""
Preprocessing design:

Numeric columns:
- Median imputation
- StandardScaler

Categorical columns:
- Most-frequent imputation
- One-hot encoding

The preprocessing object is placed inside a scikit-learn
Pipeline. Therefore it is fitted only on the training data
and then applied to the test data using transform-only logic.
This prevents test-set leakage.
""")


# ============================================================
# TASK 9
# THREE CLASSIFIERS
# ============================================================

print("\n" + "=" * 80)
print("TASK 9 - THREE CLASSIFICATION MODELS")
print("=" * 80)


models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}


fitted_models = {}


for model_name, estimator in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                estimator
            )
        ]
    )

    pipeline.fit(X_train, y_train)

    fitted_models[model_name] = pipeline

    print(f"{model_name} trained successfully.")


# ============================================================
# TASK 9
# DECISION TREE VISUALIZATION
# ============================================================

print("\n" + "=" * 80)
print("DECISION TREE VISUALIZATION")
print("=" * 80)


tree_pipeline = fitted_models["Decision Tree"]

tree_model = tree_pipeline.named_steps["model"]

tree_preprocessor = tree_pipeline.named_steps["preprocessor"]


feature_names = (
    tree_preprocessor
    .get_feature_names_out()
)


plt.figure(figsize=(24, 14))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=False,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree - Titanic Survival")

plt.tight_layout()

tree_file = output_dir / "decision_tree.png"

plt.savefig(
    tree_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Decision tree saved to:")
print(tree_file)


# ============================================================
# TASK 10
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("TASK 10 - CLASSIFICATION MODEL EVALUATION")
print("=" * 80)


results = []

roc_data = {}


for model_name, pipeline in fitted_models.items():

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("Confusion Matrix:")
    print(cm)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")


    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUC": auc
        }
    )


    # Confusion matrix image

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Survived", "Survived"]
    )

    disp.plot()

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()

    safe_name = model_name.lower().replace(
        " ",
        "_"
    )

    plt.savefig(
        output_dir / f"{safe_name}_confusion_matrix.png",
        dpi=300
    )

    plt.close()


    # ROC data

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_data[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc
    }


# ============================================================
# ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))


for model_name, values in roc_data.items():

    plt.plot(
        values["fpr"],
        values["tpr"],
        label=(
            f"{model_name} "
            f"(AUC = {values['auc']:.3f})"
        )
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curves - Titanic Classifiers")

plt.legend()

plt.tight_layout()

plt.savefig(
    output_dir / "roc_curves.png",
    dpi=300
)

plt.close()


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

classification_results = pd.DataFrame(results)

classification_results = classification_results.sort_values(
    "F1",
    ascending=False
)


print("\n" + "=" * 80)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 80)

print(
    classification_results.to_string(
        index=False
    )
)


classification_results.to_csv(
    output_dir / "classification_comparison.csv",
    index=False
)


# ============================================================
# TASK 11
# IMBALANCE HANDLING
# ============================================================

print("\n" + "=" * 80)
print("TASK 11 - CLASS IMBALANCE COMPARISON")
print("=" * 80)


print("\nOverall class balance:")

print(
    y.value_counts()
    .rename(
        index={
            0: "Not Survived",
            1: "Survived"
        }
    )
)


# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------

baseline_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


baseline_model.fit(
    X_train,
    y_train
)


baseline_pred = baseline_model.predict(
    X_test
)


# ------------------------------------------------------------
# CLASS WEIGHT BALANCED
# ------------------------------------------------------------

balanced_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


balanced_model.fit(
    X_train,
    y_train
)


balanced_pred = balanced_model.predict(
    X_test
)


# ------------------------------------------------------------
# SMOTE
# ------------------------------------------------------------

smote_model = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


smote_model.fit(
    X_train,
    y_train
)


smote_pred = smote_model.predict(
    X_test
)


# ------------------------------------------------------------
# IMBALANCE RESULTS FUNCTION
# ------------------------------------------------------------

def calculate_scores(
    name,
    actual,
    predicted
):

    return {
        "Strategy": name,

        "Precision": precision_score(
            actual,
            predicted,
            zero_division=0
        ),

        "Recall": recall_score(
            actual,
            predicted,
            zero_division=0
        ),

        "F1": f1_score(
            actual,
            predicted,
            zero_division=0
        )
    }


imbalance_results = pd.DataFrame(
    [
        calculate_scores(
            "Baseline",
            y_test,
            baseline_pred
        ),

        calculate_scores(
            "Class Weight Balanced",
            y_test,
            balanced_pred
        ),

        calculate_scores(
            "SMOTE",
            y_test,
            smote_pred
        )
    ]
)


print("\nImbalance comparison:")

print(
    imbalance_results.to_string(
        index=False
    )
)


imbalance_results.to_csv(
    output_dir / "imbalance_comparison.csv",
    index=False
)


best_imbalance = imbalance_results.loc[
    imbalance_results["F1"].idxmax()
]


print("\nBest imbalance strategy:")

print(
    best_imbalance["Strategy"]
)

print(
    f"Best F1: {best_imbalance['F1']:.4f}"
)


print("""
Conclusion:
The best imbalance strategy is selected using F1 score,
because F1 balances precision and recall. Class weighting
changes the model's learning objective, while SMOTE creates
synthetic minority-class training examples. SMOTE was applied
only after the training split and never to the test data,
preventing test-set leakage.
""")


# ============================================================
# TASK 12
# RANDOM FOREST GRID SEARCH
# ============================================================

print("\n" + "=" * 80)
print("TASK 12 - RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 80)


rf_oob_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                oob_score=True,
                bootstrap=True
            )
        )
    ]
)


param_grid = {

    "model__n_estimators": [
        100,
        200
    ],

    "model__max_depth": [
        None,
        5,
        10
    ],

    "model__max_features": [
        "sqrt",
        "log2"
    ]
}


grid_search = GridSearchCV(
    estimator=rf_oob_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


print("\nRunning GridSearchCV...")

grid_search.fit(
    X_train,
    y_train
)


print("\nBest parameters:")

print(
    grid_search.best_params_
)


best_rf_pipeline = grid_search.best_estimator_

best_rf_model = (
    best_rf_pipeline
    .named_steps["model"]
)


print("\nBest cross-validation F1:")

print(
    grid_search.best_score_
)


print("\nOOB score:")

print(
    best_rf_model.oob_score_
)


grid_results = pd.DataFrame(
    grid_search.cv_results_
)

grid_results.to_csv(
    output_dir / "random_forest_grid_search.csv",
    index=False
)


# ============================================================
# TASK 13
# REGRESSION SIDE TASK
# PREDICT FARE
# ============================================================

print("\n" + "=" * 80)
print("TASK 13 - FARE REGRESSION")
print("=" * 80)


# We predict fare using other available passenger features.
# Fare itself is NOT included as an input feature.

regression_features_numeric = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]


regression_features_categorical = [
    "sex",
    "embarked"
]


regression_features = (
    regression_features_numeric
    +
    regression_features_categorical
)


X_reg = df[regression_features]

y_reg = df["fare"]


X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)


regression_numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


regression_categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            regression_numeric_pipeline,
            regression_features_numeric
        ),

        (
            "categorical",
            regression_categorical_pipeline,
            regression_features_categorical
        )
    ]
)


regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            regression_preprocessor
        ),

        (
            "model",
            LinearRegression()
        )
    ]
)


print("\nTraining linear regression...")

regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)


regression_predictions = regression_pipeline.predict(
    X_reg_test
)


mae = mean_absolute_error(
    y_reg_test,
    regression_predictions
)


rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        regression_predictions
    )
)


r2 = r2_score(
    y_reg_test,
    regression_predictions
)


# Adjusted R2
n = len(y_reg_test)

p = (
    regression_pipeline
    .named_steps["preprocessor"]
    .transform(X_reg_test)
    .shape[1]
)


adjusted_r2 = (
    1
    -
    (
        (1 - r2)
        *
        (n - 1)
        /
        (n - p - 1)
    )
)


print("\nRegression metrics:")

print(f"MAE         : {mae:.4f}")
print(f"RMSE        : {rmse:.4f}")
print(f"R2          : {r2:.4f}")
print(f"Adjusted R2 : {adjusted_r2:.4f}")


# ============================================================
# RESIDUAL PLOT
# ============================================================

residuals = (
    y_reg_test
    -
    regression_predictions
)


plt.figure(figsize=(8, 6))

plt.scatter(
    regression_predictions,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")

plt.ylabel("Residual")

plt.title(
    "Linear Regression Residual Plot"
)

plt.tight_layout()

plt.savefig(
    output_dir / "regression_residual_plot.png",
    dpi=300
)

plt.close()


print("""
Heteroscedasticity conclusion:
The residual plot should be inspected for whether the spread
of residuals changes systematically as predicted fare
increases. A widening or narrowing funnel-shaped pattern
would indicate heteroscedasticity; a roughly random and
constant spread would suggest no strong evidence of
heteroscedasticity.
""")


# ============================================================
# TASK 14
# FINAL MODEL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("TASK 14 - FINAL MODEL COMPARISON")
print("=" * 80)


final_classification_table = classification_results.copy()


final_classification_table[
    "Metric Group"
] = "Classification"


regression_row = pd.DataFrame(
    [
        {
            "Model": "Linear Regression",
            "Accuracy": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "AUC": np.nan,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted_R2": adjusted_r2,
            "Metric Group": "Regression"
        }
    ]
)


for column in [
    "MAE",
    "RMSE",
    "R2",
    "Adjusted_R2"
]:

    if column not in final_classification_table.columns:

        final_classification_table[column] = np.nan


final_table = pd.concat(
    [
        final_classification_table,
        regression_row
    ],
    ignore_index=True
)


print("\nFinal comparison table:")

print(
    final_table.to_string(
        index=False
    )
)


final_table.to_csv(
    output_dir / "final_model_comparison.csv",
    index=False
)


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

best_classifier = classification_results.iloc[0]


print("\n" + "=" * 80)
print("FINAL RECOMMENDATION")
print("=" * 80)


print(
    f"""
Recommended classifier: {best_classifier['Model']}

It achieved the highest F1 score among the three evaluated
classifiers, with F1 = {best_classifier['F1']:.4f} and
AUC = {best_classifier['AUC']:.4f}. Its accuracy was
{best_classifier['Accuracy']:.4f}, precision was
{best_classifier['Precision']:.4f}, and recall was
{best_classifier['Recall']:.4f}.

The F1 score is particularly useful here because the Titanic
target is imbalanced and both false positives and false
negatives matter. The ROC AUC also indicates how well the
classifier separates survivors from non-survivors across
different probability thresholds.

Therefore, based on the measured test-set performance, the
classifier with the highest F1 score is recommended for
deployment.
"""
)


# ============================================================
# TASK 15
# SAVE COMPLETE FITTED PIPELINE
# ============================================================

print("\n" + "=" * 80)
print("TASK 15 - SAVE COMPLETE PIPELINE")
print("=" * 80)


best_classifier_name = best_classifier["Model"]

best_classifier_pipeline = fitted_models[
    best_classifier_name
]


pipeline_file = BASE_DIR / "best_titanic_pipeline.joblib"


joblib.dump(
    best_classifier_pipeline,
    pipeline_file
)


print("\nComplete pipeline saved to:")

print(pipeline_file)


# ============================================================
# RELOAD SAVED PIPELINE
# ============================================================

print("\nReloading saved pipeline...")

loaded_pipeline = joblib.load(
    pipeline_file
)


# Raw new passenger data
# No manual preprocessing is performed.

raw_example = pd.DataFrame(
    [
        {
            "pclass": 1,
            "sex": "female",
            "age": 30,
            "sibsp": 0,
            "parch": 0,
            "fare": 80.0,
            "embarked": "C"
        }
    ]
)


loaded_prediction = loaded_pipeline.predict(
    raw_example
)


loaded_probability = (
    loaded_pipeline
    .predict_proba(raw_example)[0][1]
)


print("\nRaw input:")

print(raw_example)


print("\nPrediction after reloading:")

print(
    "Predicted survival:",
    int(loaded_prediction[0])
)


print(
    "Survival probability:",
    round(
        loaded_probability,
        4
    )
)


print("""
The saved object contains the preprocessing steps and the
classifier together. Therefore raw passenger data can be
passed directly to the loaded pipeline without manually
imputing, encoding, or scaling the input.
""")


# ============================================================
# SAVE WRITTEN SUMMARY
# ============================================================

summary_file = output_dir / "model_summary.txt"


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ZEPTO MODULE 2 - MODELING SUMMARY\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        "CLASSIFICATION RESULTS\n"
    )

    f.write(
        classification_results
        .to_string(index=False)
    )

    f.write("\n\n")

    f.write(
        "IMBALANCE COMPARISON\n"
    )

    f.write(
        imbalance_results
        .to_string(index=False)
    )

    f.write("\n\n")

    f.write(
        "RANDOM FOREST BEST PARAMETERS\n"
    )

    f.write(
        str(grid_search.best_params_)
    )

    f.write("\n\n")

    f.write(
        "RANDOM FOREST OOB SCORE\n"
    )

    f.write(
        str(best_rf_model.oob_score_)
    )

    f.write("\n\n")

    f.write(
        "REGRESSION RESULTS\n"
    )

    f.write(
        f"MAE: {mae:.4f}\n"
    )

    f.write(
        f"RMSE: {rmse:.4f}\n"
    )

    f.write(
        f"R2: {r2:.4f}\n"
    )

    f.write(
        f"Adjusted R2: {adjusted_r2:.4f}\n"
    )

    f.write("\n")

    f.write(
        "RECOMMENDED CLASSIFIER\n"
    )

    f.write(
        best_classifier_name
    )


print("\nSummary saved to:")

print(summary_file)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 80)
print("MODULE 2 PART B COMPLETED SUCCESSFULLY!")
print("=" * 80)

print("\nGenerated files are inside:")

print(output_dir)

print("\nMain model artifact:")

print(pipeline_file)