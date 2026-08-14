import sqlite3
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "books.db"
CSV_PATH = ROOT / "cleaned_books.csv"


def get_soup(url):
    response = requests.get(
        url,
        timeout=20,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    print(f"HTTP {response.status_code}: {url}")

    response.raise_for_status()

    return BeautifulSoup(
        response.text,
        "html.parser"
    )


def scrape_category(category_name, category_url):
    print(f"\nScraping category: {category_name}")

    soup = get_soup(category_url)

    books = []

    for card in soup.select("article.product_pod"):

        title_tag = card.select_one("h3 a")
        price_tag = card.select_one(".price_color")
        rating_tag = card.select_one("p.star-rating")
        availability_tag = card.select_one(".availability")

        title = (
            title_tag.get("title", "").strip()
            if title_tag
            else ""
        )

        price = (
            price_tag.get_text(strip=True)
            if price_tag
            else ""
        )

        availability = (
            availability_tag.get_text(
                " ",
                strip=True
            )
            if availability_tag
            else ""
        )

        rating_text = ""

        if rating_tag:

            classes = rating_tag.get(
                "class",
                []
            )

            for value in classes:

                if value in RATING_MAP:
                    rating_text = value
                    break

        books.append({
            "title": title,
            "price": price,
            "star_rating": rating_text,
            "availability": availability,
            "category": category_name
        })

    return books


def scrape_books():

    categories = {
        "Travel": (
            BASE_URL
            + "catalogue/category/books/travel_2/index.html"
        ),

        "Mystery": (
            BASE_URL
            + "catalogue/category/books/mystery_3/index.html"
        ),

        "Historical Fiction": (
            BASE_URL
            + "catalogue/category/books/"
            + "historical-fiction_4/index.html"
        ),

        "Science Fiction": (
            BASE_URL
            + "catalogue/category/books/"
            + "science-fiction_16/index.html"
        ),
    }

    all_books = []

    for category_name, url in categories.items():

        books = scrape_category(
            category_name,
            url
        )

        all_books.extend(books)

        print(
            f"Books found: {len(books)}"
        )

    return pd.DataFrame(all_books)


def clean_data(df):
    """
    Clean scraped data and create the required typed columns.

    Numeric parsing failures are handled using median imputation.
    """

    df = df.copy()

    # ========================================================
    # PRICE: GBP -> FLOAT
    # ========================================================

    # Extract the numeric portion directly.
    # This is more robust than relying only on replacing "£".
    df["price_gbp"] = (
        df["price"]
        .astype(str)
        .str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
    )

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    # Median imputation for numeric parsing failures.
    if df["price_gbp"].isna().any():

        median_price = df["price_gbp"].median()

        if pd.isna(median_price):
            raise ValueError(
                "All price values failed to parse. "
                "Cannot calculate median price."
            )

        df["price_gbp"] = (
            df["price_gbp"]
            .fillna(median_price)
        )

    # ========================================================
    # STAR RATING: TEXT -> INTEGER
    # ========================================================

    df["rating"] = (
        df["star_rating"]
        .astype(str)
        .map(RATING_MAP)
    )

    if df["rating"].isna().any():

        median_rating = df["rating"].median()

        if pd.isna(median_rating):
            raise ValueError(
                "All rating values failed to parse."
            )

        df["rating"] = (
            df["rating"]
            .fillna(median_rating)
        )

    df["rating"] = (
        df["rating"]
        .round()
        .astype(int)
    )

    # ========================================================
    # AVAILABILITY: TEXT -> BOOLEAN
    # ========================================================

    availability = (
        df["availability"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["in_stock"] = (
        availability
        .str.contains(
            "in stock",
            na=False
        )
    )

    df["in_stock"] = (
        df["in_stock"]
        .fillna(False)
        .astype(bool)
    )

    # ========================================================
    # GBP -> INR
    # ========================================================

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    # ========================================================
    # FINAL TYPE ENFORCEMENT
    # ========================================================

    df["price_gbp"] = (
        pd.to_numeric(
            df["price_gbp"],
            errors="coerce"
        )
        .astype(float)
    )

    df["price_inr"] = (
        pd.to_numeric(
            df["price_inr"],
            errors="coerce"
        )
        .astype(float)
    )

    # Final safety check.
    # The database has NOT NULL constraints, so do not allow
    # missing numeric values to reach SQLite.
    if df["price_gbp"].isna().any():

        median_price = df["price_gbp"].median()

        df["price_gbp"] = (
            df["price_gbp"]
            .fillna(median_price)
        )

    if df["price_inr"].isna().any():

        df["price_inr"] = (
            df["price_gbp"] * GBP_TO_INR
        ).round(2)

    # Final validation.
    required_numeric = [
        "price_gbp",
        "price_inr",
        "rating"
    ]

    for column in required_numeric:

        if df[column].isna().any():

            raise ValueError(
                f"Missing values remain in {column}"
            )

    return df[
        [
            "title",
            "price",
            "price_gbp",
            "star_rating",
            "rating",
            "availability",
            "in_stock",
            "price_inr",
            "category"
        ]
    ]


def create_database(df):

    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(
        DB_PATH
    )

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    # -----------------------------
    # CATEGORIES TABLE
    # -----------------------------

    conn.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
        """
    )

    # -----------------------------
    # BOOKS TABLE
    # -----------------------------

    conn.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY(category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    # Insert categories

    category_names = sorted(
        df["category"].unique()
    )

    conn.executemany(
        """
        INSERT INTO categories(category_name)
        VALUES (?)
        """,
        [
            (name,)
            for name in category_names
        ]
    )

    # Category lookup

    lookup = dict(
        conn.execute(
            """
            SELECT
                category_name,
                category_id
            FROM categories
            """
        ).fetchall()
    )

    # Insert books

    rows = []

    for row in df.itertuples(
        index=False
    ):

        rows.append(
            (
                row.title,
                float(row.price_gbp),
                float(row.price_inr),
                int(row.rating),
                int(row.in_stock),
                int(lookup[row.category])
            )
        )

    conn.executemany(
        """
        INSERT INTO books
        (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows
    )

    conn.commit()

    conn.close()


def main():

    print("=" * 60)
    print("ZEpto MODULE 1 - DATA PIPELINE")
    print("=" * 60)

    raw_df = scrape_books()

    print(
        f"\nTotal books scraped: "
        f"{len(raw_df)}"
    )

    print(
        f"Categories found: "
        f"{raw_df['category'].nunique()}"
    )

    if len(raw_df) < 60:
        raise ValueError(
            "ERROR: Fewer than 60 books scraped."
        )

    if raw_df["category"].nunique() < 3:
        raise ValueError(
            "ERROR: Fewer than 3 categories."
        )

    cleaned_df = clean_data(
        raw_df
    )

    cleaned_df.to_csv(
        CSV_PATH,
        index=False
    )

    create_database(
        cleaned_df
    )

    print("\nSUCCESS!")

    print(
        f"Books: {len(cleaned_df)}"
    )

    print(
        f"Categories: "
        f"{cleaned_df['category'].nunique()}"
    )

    print(
        "\nFixed conversion:"
    )

    print(
        "1 GBP = 105.50 INR"
    )

    print(
        "\nData types:"
    )

    print(
        cleaned_df[
            [
                "price_gbp",
                "rating",
                "in_stock",
                "price_inr"
            ]
        ].dtypes
    )

    print(
        f"\nCreated database:"
        f"\n{DB_PATH}"
    )

    print(
        f"\nCreated CSV:"
        f"\n{CSV_PATH}"
    )


if __name__ == "__main__":
    main()