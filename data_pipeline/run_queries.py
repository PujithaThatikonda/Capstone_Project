import sqlite3
from pathlib import Path

import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DB_PATH = ROOT / "books.db"
OUTPUT_PATH = ROOT / "query_outputs.txt"


# ============================================================
# CONNECT TO DATABASE
# ============================================================

conn = sqlite3.connect(DB_PATH)

# Enable foreign-key enforcement
conn.execute("PRAGMA foreign_keys = ON")


# ============================================================
# HELPER FUNCTION
# ============================================================

def run_query(query_number, query, conn):

    print("\n" + "=" * 80)
    print(f"QUERY {query_number}")
    print("=" * 80)

    print("\nSQL:")
    print(query)

    result = pd.read_sql_query(
        query,
        conn
    )

    print("\nOUTPUT:")
    print(result.to_string(index=False))

    return result


# ============================================================
# QUERY 1
# SELECT + WHERE
# ============================================================

query1 = """
SELECT
    title,
    price_gbp,
    rating,
    in_stock
FROM books
WHERE rating >= 4
ORDER BY rating DESC;
"""


# ============================================================
# QUERY 2
# ORDER BY + LIMIT
# ============================================================

query2 = """
SELECT
    title,
    price_gbp,
    price_inr,
    rating
FROM books
ORDER BY price_gbp DESC
LIMIT 10;
"""


# ============================================================
# QUERY 3
# DISTINCT
# ============================================================

query3 = """
SELECT DISTINCT
    category_name
FROM categories
ORDER BY category_name;
"""


# ============================================================
# QUERY 4
# BETWEEN
# ============================================================

query4 = """
SELECT
    title,
    price_gbp,
    rating
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp;
"""


# ============================================================
# QUERY 5
# IN
# ============================================================

query5 = """
SELECT
    title,
    rating,
    price_gbp
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC, price_gbp DESC;
"""


# ============================================================
# QUERY 6
# JOIN
# ============================================================

query6 = """
SELECT
    b.title,
    c.category_name,
    b.price_gbp,
    b.price_inr,
    b.rating,
    b.in_stock
FROM books AS b
JOIN categories AS c
    ON b.category_id = c.category_id
ORDER BY b.rating DESC, b.price_gbp DESC
LIMIT 10;
"""


# ============================================================
# EXECUTE QUERIES
# ============================================================

results = {}

results["query1"] = run_query(
    1,
    query1,
    conn
)

results["query2"] = run_query(
    2,
    query2,
    conn
)

results["query3"] = run_query(
    3,
    query3,
    conn
)

results["query4"] = run_query(
    4,
    query4,
    conn
)

results["query5"] = run_query(
    5,
    query5,
    conn
)

results["query6"] = run_query(
    6,
    query6,
    conn
)


# ============================================================
# READ SQL RESULTS INTO PANDAS
# ============================================================

print("\n" + "=" * 80)
print("PANDAS pd.read_sql() RESULTS")
print("=" * 80)

pd_sql_result_1 = pd.read_sql(
    query1,
    conn
)

pd_sql_join = pd.read_sql(
    query6,
    conn
)

print("\nResult loaded using pd.read_sql():")
print(pd_sql_result_1.to_string(index=False))

print("\nJOIN result loaded using pd.read_sql():")
print(pd_sql_join.to_string(index=False))


# ============================================================
# LOAD RAW TABLES INTO PANDAS
# ============================================================

books_df = pd.read_sql(""" SELECT * FROM books """,conn)

categories_df = pd.read_sql(
    """
    SELECT * FROM categories
    """,
    conn
)

# ============================================================
# REPRODUCE JOIN USING pd.merge()
# ============================================================

print("\n" + "=" * 80)
print("PANDAS pd.merge() JOIN")
print("=" * 80)

merged_df = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

# Select the same columns as SQL JOIN
pd_merge_join = merged_df[
    [
        "title",
        "category_name",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock"
    ]
].sort_values(
    by=["rating", "price_gbp"],
    ascending=[False, False]
).head(10)


print(
    pd_merge_join.to_string(
        index=False
    )
)


# ============================================================
# COMPARE SQL JOIN AND PANDAS MERGE
# ============================================================

sql_compare = pd_sql_join.reset_index(
    drop=True
)

merge_compare = pd_merge_join.reset_index(
    drop=True
)

# Ensure identical column ordering and values
sql_compare["in_stock"] = (
    sql_compare["in_stock"]
    .astype(bool)
)

merge_compare["in_stock"] = (
    merge_compare["in_stock"]
    .astype(bool)
)

equivalent = sql_compare.equals(
    merge_compare
)

print("\n" + "=" * 80)
print("JOIN EQUIVALENCE CHECK")
print("=" * 80)

print(
    "SQL JOIN and pandas.merge() "
    f"produce equivalent results: {equivalent}"
)


# ============================================================
# SAVE ALL SQL OUTPUTS
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "ZEpto MODULE 1 - SQL QUERY OUTPUTS\n"
    )

    file.write(
        "=" * 80 + "\n\n"
    )

    queries = [
        (1, query1, results["query1"]),
        (2, query2, results["query2"]),
        (3, query3, results["query3"]),
        (4, query4, results["query4"]),
        (5, query5, results["query5"]),
        (6, query6, results["query6"]),
    ]

    for number, query, result in queries:

        file.write(
            "=" * 80 + "\n"
        )

        file.write(
            f"QUERY {number}\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        file.write(
            "SQL:\n"
        )

        file.write(
            query.strip() + "\n\n"
        )

        file.write(
            "OUTPUT:\n"
        )

        file.write(
            result.to_string(index=False)
            + "\n\n"
        )

    file.write(
        "=" * 80 + "\n"
    )

    file.write(
        "PANDAS JOIN EQUIVALENCE\n"
    )

    file.write(
        "=" * 80 + "\n\n"
    )

    file.write(
        "pd.read_sql() JOIN:\n"
    )

    file.write(
        pd_sql_join.to_string(
            index=False
        )
        + "\n\n"
    )

    file.write(
        "pd.merge() JOIN:\n"
    )

    file.write(
        pd_merge_join.to_string(
            index=False
        )
        + "\n\n"
    )

    file.write(
        "Equivalent: "
        + str(equivalent)
        + "\n"
    )


# ============================================================
# CLOSE DATABASE
# ============================================================

conn.close()


print("\n" + "=" * 80)
print("MODULE 1 SQL STEP COMPLETE")
print("=" * 80)

print(
    f"\nSaved query output to:\n"
    f"{OUTPUT_PATH}"
)