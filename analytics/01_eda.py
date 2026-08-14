
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# ============================================================
# ZEPTO MODULE 2 - PART A
# TITANIC: PROFILING, CLEANING AND EDA
# ============================================================

print("=" * 70)
print("ZEPTO MODULE 2 - TITANIC ANALYTICS PIPELINE")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

input_file = BASE_DIR / "titanic.csv"

if not input_file.exists():
    print("ERROR: titanic.csv not found.")
    print("Expected location:")
    print(input_file)
    raise SystemExit

df = pd.read_csv(input_file)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# REQUIRED PROFILING
# ============================================================

print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print(df.shape)


print("\n" + "=" * 70)
print("DATASET INFO")
print("=" * 70)

df.info()


print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(df.describe())


# ============================================================
# 2. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_counts = df.isnull().sum()

print(missing_counts)


print("\n" + "=" * 70)
print("MISSING VALUE PERCENTAGES")
print("=" * 70)

missing_percentages = (df.isnull().mean() * 100).round(2)

missing_table = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_percentage": missing_percentages
})

missing_table = missing_table[
    missing_table["missing_count"] > 0
]

print(missing_table)


# ============================================================
# 3. CLEANING
# ============================================================

print("\n" + "=" * 70)
print("CLEANING DECISIONS")
print("=" * 70)

print("""
Cleaning rule:

1. Less than 5% missing:
   Drop rows containing missing values.

2. 5% to 30% missing:
   Impute missing values.

3. More than 30% missing:
   Drop the column if it is not required.
""")


# ------------------------------------------------------------
# AGE
# 19.87% missing -> 5%-30% -> median imputation
# ------------------------------------------------------------

age_missing_pct = df["age"].isnull().mean() * 100

print(f"age missing percentage: {age_missing_pct:.2f}%")

age_median = df["age"].median()

df["age"] = df["age"].fillna(age_median)

print(
    f"age strategy: median imputation "
    f"because missingness is {age_missing_pct:.2f}%."
)

print(f"Age median used: {age_median:.2f}")


# ------------------------------------------------------------
# EMBARKED
# 0.22% missing -> <5% -> drop rows
# ------------------------------------------------------------

embarked_missing_pct = df["embarked"].isnull().mean() * 100

print(f"\nembarked missing percentage: {embarked_missing_pct:.2f}%")

df = df.dropna(subset=["embarked"])

print(
    "embarked strategy: dropped rows because missingness "
    f"is only {embarked_missing_pct:.2f}% (<5%)."
)


# ------------------------------------------------------------
# EMBARK_TOWN
# 0.22% missing -> <5% -> drop rows
# ------------------------------------------------------------

embark_town_missing_pct = df["embark_town"].isnull().mean() * 100

print(
    f"\nembark_town missing percentage: "
    f"{embark_town_missing_pct:.2f}%"
)

df = df.dropna(subset=["embark_town"])

print(
    "embark_town strategy: dropped rows because missingness "
    f"is only {embark_town_missing_pct:.2f}% (<5%)."
)


# ------------------------------------------------------------
# DECK
# 77.22% missing -> >30% -> drop column
# ------------------------------------------------------------

deck_missing_pct = missing_percentages["deck"]

print(f"\ndeck missing percentage: {deck_missing_pct:.2f}%")

df = df.drop(columns=["deck"])

print(
    "deck strategy: column dropped because "
    f"{deck_missing_pct:.2f}% of values are missing (>30%), "
    "making reliable imputation inappropriate."
)


print("\nCleaned dataset shape:", df.shape)


# ============================================================
# SAVE CLEANED DATA
# ============================================================

cleaned_file = BASE_DIR / "cleaned_titanic.csv"

df.to_csv(cleaned_file, index=False)

print("\nCleaned dataset saved to:")
print(cleaned_file)


# ============================================================
# 4. FARE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FARE STATISTICS")
print("=" * 70)

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

print(f"Mean   : {fare_mean:.4f}")
print(f"Median : {fare_median:.4f}")
print(f"Mode   : {fare_mode:.4f}")


if fare_mean > fare_median > fare_mode:
    fare_skew = "right-skewed"
elif fare_mean < fare_median < fare_mode:
    fare_skew = "left-skewed"
else:
    fare_skew = "approximately symmetric"

print("\nFare distribution:", fare_skew)

print("""
Interpretation:
The mean fare is substantially greater than the median,
and the mode is much lower. This ordering indicates a
right-skewed distribution caused by a relatively small
number of passengers paying very high fares.
""")


# ============================================================
# 5. IQR OUTLIERS
# ============================================================

print("\n" + "=" * 70)
print("IQR OUTLIER ANALYSIS")
print("=" * 70)


def count_iqr_outliers(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
    ]

    return len(outliers), lower_bound, upper_bound


age_outliers, age_lower, age_upper = count_iqr_outliers(df["age"])

fare_outliers, fare_lower, fare_upper = count_iqr_outliers(df["fare"])


print(f"Age outliers : {age_outliers}")
print(f"Age bounds   : [{age_lower:.2f}, {age_upper:.2f}]")

print()

print(f"Fare outliers: {fare_outliers}")
print(f"Fare bounds  : [{fare_lower:.2f}, {fare_upper:.2f}]")


# ============================================================
# 6. BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("SURVIVAL RATE BY SEX")
print("=" * 70)

female_mask = df["sex"] == "female"
male_mask = df["sex"] == "male"

female_survival = df.loc[female_mask, "survived"].mean()
male_survival = df.loc[male_mask, "survived"].mean()

print(f"Female survival rate: {female_survival:.4f}")
print(f"Male survival rate  : {male_survival:.4f}")


print("\n" + "=" * 70)
print("SURVIVAL RATE BY PCLASS")
print("=" * 70)

for pclass in sorted(df["pclass"].unique()):

    pclass_mask = df["pclass"] == pclass

    survival_rate = df.loc[
        pclass_mask,
        "survived"
    ].mean()

    print(
        f"Class {pclass}: "
        f"{survival_rate:.4f}"
    )


# ============================================================
# SEX + PCLASS
# ============================================================

print("\n" + "=" * 70)
print("SURVIVAL RATE BY SEX AND PCLASS")
print("=" * 70)

for sex in ["female", "male"]:

    for pclass in sorted(df["pclass"].unique()):

        combined_mask = (
            (df["sex"] == sex) &
            (df["pclass"] == pclass)
        )

        rate = df.loc[
            combined_mask,
            "survived"
        ].mean()

        print(
            f"{sex.capitalize()}, "
            f"Class {pclass}: "
            f"{rate:.4f}"
        )


# ============================================================
# 7. EXACT SIX-COLUMN CORRELATION MATRIX
# ============================================================

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = df[
    correlation_columns
].corr()

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

print(correlation_matrix.round(3))


# ============================================================
# FIND TWO STRONGEST CORRELATIONS
# ============================================================

pairs = []

for i in range(len(correlation_columns)):

    for j in range(i + 1, len(correlation_columns)):

        col1 = correlation_columns[i]
        col2 = correlation_columns[j]

        correlation = correlation_matrix.loc[
            col1,
            col2
        ]

        pairs.append(
            (
                col1,
                col2,
                correlation,
                abs(correlation)
            )
        )


pairs = sorted(
    pairs,
    key=lambda x: x[3],
    reverse=True
)

print("\n" + "=" * 70)
print("TWO STRONGEST CORRELATIONS")
print("=" * 70)

for pair in pairs[:2]:

    print(
        f"{pair[0]} vs {pair[1]}: "
        f"{pair[2]:.4f}"
    )


# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Titanic Numeric Feature Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    BASE_DIR / "correlation_heatmap.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. CHART 1
# SURVIVAL BY SEX
# ============================================================

plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="sex",
    y="survived"
)

plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")

plt.tight_layout()

plt.savefig(
    BASE_DIR / "01_survival_by_sex.png",
    dpi=300
)

plt.show()


print("""
Chart 1 interpretation:
Women had a substantially higher survival rate than men.
This suggests that sex was an important factor associated
with survival on the Titanic.
""")


# ============================================================
# 9. CHART 2
# SURVIVAL BY PCLASS
# ============================================================

plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived"
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.tight_layout()

plt.savefig(
    BASE_DIR / "02_survival_by_class.png",
    dpi=300
)

plt.show()


print("""
Chart 2 interpretation:
Passengers in higher classes had higher survival rates.
First-class passengers were substantially more likely to
survive than third-class passengers, indicating that
passenger class was strongly associated with survival.
""")


# ============================================================
# 10. CHART 3
# AGE DISTRIBUTION BY SURVIVAL
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="survived",
    y="age"
)

plt.title("Age Distribution by Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")

plt.tight_layout()

plt.savefig(
    BASE_DIR / "03_age_by_survival.png",
    dpi=300
)

plt.show()


print("""
Chart 3 interpretation:
The age distributions of survivors and non-survivors
overlap considerably, but younger passengers appear more
represented among survivors. Age therefore provides useful
information, although it is not as strongly separated as sex.
""")


# ============================================================
# 11. CHART 4
# FARE BY CLASS AND SURVIVAL
# ============================================================

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="pclass",
    y="fare",
    hue="survived"
)

plt.title("Fare Distribution by Class and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    BASE_DIR / "04_fare_class_survival.png",
    dpi=300
)

plt.show()


print("""
Chart 4 interpretation:
First-class passengers generally paid much higher fares
than passengers in lower classes. Within classes, fare
differences also provide information about survival,
suggesting that socioeconomic position was related to
survival probability.
""")


# ============================================================
# 12. CHART 5
# SEX + PCLASS
# ============================================================

plt.figure(figsize=(9, 6))

sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.tight_layout()

plt.savefig(
    BASE_DIR / "05_sex_class_survival.png",
    dpi=300
)

plt.show()


print("""
Chart 5 interpretation:
Combining sex and passenger class reveals an even clearer
survival pattern. Female passengers generally had higher
survival rates across classes, while male passengers,
particularly those in third class, had much lower survival.
""")


# ============================================================
# 13. EXPLORATORY STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZATION CHECK")
print("=" * 70)

standardizer = StandardScaler()

df_standardized = df.copy()

df_standardized[
    ["age_z", "fare_z"]
] = standardizer.fit_transform(
    df[["age", "fare"]]
)


print("\nBefore standardization:")

print(
    df[["age", "fare"]].agg(
        ["mean", "std"]
    )
)


print("\nAfter standardization:")

print(
    df_standardized[
        ["age_z", "fare_z"]
    ].agg(
        ["mean", "std"]
    )
)


print("""
Interpretation:
Before standardization, age and fare are measured on
different scales. After z-score standardization, both
variables have means approximately equal to 0 and standard
deviations approximately equal to 1.
""")


# ============================================================
# 14. SAVE FINAL CLEANED DATA
# ============================================================

final_file = BASE_DIR / "cleaned_titanic.csv"

df.to_csv(
    final_file,
    index=False
)

print("\nFinal cleaned dataset saved:")
print(final_file)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(f"Original rows : 891")
print(f"Cleaned rows  : {len(df)}")
print(f"Columns       : {len(df.columns)}")
print("Missing Values ")
print("-"*40)
print(df.isnull().sum())

print("\nFiles created:")

print("1. cleaned_titanic.csv")
print("2. correlation_heatmap.png")
print("3. 01_survival_by_sex.png")
print("4. 02_survival_by_class.png")
print("5. 03_age_by_survival.png")
print("6. 04_fare_class_survival.png")
print("7. 05_sex_class_survival.png")

print("\nModule 2 Part A completed successfully!")