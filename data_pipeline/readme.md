# Module 1 — Data Pipeline

## 1. Project Overview

This project demonstrates a complete data pipeline using book data from **Books to Scrape**.

The pipeline performs these steps:

**Scrape → Clean → Convert → Store → Query → Analyze**

The data source is:

**Books to Scrape:** http://books.toscrape.com/

The website is a public scraping-practice website and does not require a login or API key.

---

## 2. Tools and Libraries Used

The project uses:

* Python
* `requests` — to download web pages
* `BeautifulSoup` — to extract book information from HTML
* `pandas` — to clean and analyze the data
* `sqlite3` — to create and query the SQLite database

Install the required libraries with:

```bash
pip install requests beautifulsoup4 pandas lxml
```

---

## 3. Data Collected

The scraper collects books from four categories:

1. Travel
2. Mystery
3. Historical Fiction
4. Science Fiction

The final dataset contains **67 books across 4 categories**.

For every book, the scraper collects:

* Title
* Price
* Star rating
* Availability
* Category

---

## 4. Data Cleaning

The scraped data is cleaned before being stored in the database.

### Price

The original price contains the `£` currency symbol.

For example:

```text
£51.77
```

The currency symbol is removed and the value is converted to a floating-point number.

The cleaned column is:

```text
price_gbp
```

### Star Rating

The text ratings are converted to integers:

| Original | Numeric |
| -------- | ------: |
| One      |       1 |
| Two      |       2 |
| Three    |       3 |
| Four     |       4 |
| Five     |       5 |

The cleaned column is:

```text
rating
```

### Availability

The availability text is converted into a Boolean value.

For example:

```text
In stock
```

becomes:

```text
True
```

The cleaned column is:

```text
in_stock
```

### Missing or Invalid Numeric Values

If a numeric value cannot be parsed, the pipeline converts it to a missing value and uses **median imputation** for the numeric column.

This prevents the pipeline from crashing because of unexpected scraped values.

---

## 5. Currency Conversion

The project requires a fixed conversion rate.

The rate used is:

```text
1 GBP = 105.50 INR
```

This is a fixed project-defined rate and is **not a live exchange rate**.

The INR price is calculated as:

```text
price_inr = price_gbp × 105.50
```

For example:

```text
£10 × 105.50 = ₹1,055.00
```

No currency API is required.

---

## 6. Database Design

The cleaned data is stored in a SQLite database called:

```text
books.db
```

The database contains two normalized tables.

### categories

```text
category_id
category_name
```

`category_id` is the primary key.

### books

```text
book_id
title
price_gbp
price_inr
rating
in_stock
category_id
```

`book_id` is the primary key.

`category_id` is a foreign key referencing the `categories` table.

The relationship is:

```text
categories
    |
    | category_id
    |
    ↓
books
```

This separates category information from book information and avoids repeating category names unnecessarily.

---

## 7. Running the Project

### Step 1 — Run the scraper

From the project folder:

```bash
python scrape_and_load.py
```

This will:

1. Download the book pages.
2. Scrape the book information.
3. Clean the data.
4. Convert GBP prices to INR.
5. Create the SQLite database.
6. Save the cleaned CSV file.

The generated files are:

```text
books.db
cleaned_books.csv
```

---

### Step 2 — Run the SQL queries

Run:

```bash
python run_queries.py
```

This executes the required SQL queries and saves their results in:

```text
query_outputs.txt
```

---

## 8. SQL Queries

The project contains six SQL queries.

### Query 1 — SELECT and WHERE

Find books with a rating of 4 or higher.

```sql
SELECT
    title,
    price_gbp,
    rating,
    in_stock
FROM books
WHERE rating >= 4
ORDER BY rating DESC;
```

This demonstrates:

* `SELECT`
* `WHERE`
* `ORDER BY`

---

### Query 2 — ORDER BY and LIMIT

Find the 10 most expensive books.

```sql
SELECT
    title,
    price_gbp,
    price_inr,
    rating
FROM books
ORDER BY price_gbp DESC
LIMIT 10;
```

This demonstrates:

* `ORDER BY`
* `LIMIT`

---

### Query 3 — DISTINCT

List the different book categories.

```sql
SELECT DISTINCT
    category_name
FROM categories
ORDER BY category_name;
```

This demonstrates:

* `DISTINCT`

---

### Query 4 — BETWEEN

Find books with prices between £20 and £40.

```sql
SELECT
    title,
    price_gbp,
    rating
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp;
```

This demonstrates:

* `BETWEEN`

---

### Query 5 — IN

Find books with ratings of 4 or 5.

```sql
SELECT
    title,
    rating,
    price_gbp
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC, price_gbp DESC;
```

This demonstrates:

* `IN`

---

### Query 6 — JOIN

Join the books and categories tables.

```sql
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
```

This demonstrates the required:

* Primary key / foreign key relationship
* `JOIN`
* `ORDER BY`
* `LIMIT`

---

## 9. Pandas Analysis

The SQL query results are also read into pandas using:

```python
pd.read_sql()
```

The books and categories tables are separately loaded into pandas DataFrames.

The same JOIN operation is then reproduced without SQL using:

```python
pd.merge()
```

The SQL JOIN result and pandas merge result are compared.

The project checks that:

```text
SQL JOIN and pandas.merge() produce equivalent results: True
```

This demonstrates that the same relational operation can be reproduced using pandas.

---

## 10. Project Files

The final project contains:

```text
data_pipeline/
│
├── scrape_and_load.py
├── run_queries.py
├── books.db
├── cleaned_books.csv
├── query_outputs.txt
└── README.md
```

### File descriptions

| File                 | Purpose                                      |
| -------------------- | -------------------------------------------- |
| `scrape_and_load.py` | Scrapes, cleans, converts and loads the data |
| `run_queries.py`     | Runs SQL queries and pandas analysis         |
| `books.db`           | SQLite database                              |
| `cleaned_books.csv`  | Cleaned dataset                              |
| `query_outputs.txt`  | SQL queries and their outputs                |
| `README.md`          | Project documentation                        |

---

## 11. Final Results

The pipeline successfully produced:

```text
Books scraped: 67
Categories: 4
Currency rate: 1 GBP = 105.50 INR
Database: books.db
Cleaned data: cleaned_books.csv
SQL output: query_outputs.txt
```

The pipeline can be recreated from scratch by running:

```bash
python scrape_and_load.py
python run_queries.py
```

This completes the required Module 1 data pipeline: **scraping, cleaning, currency conversion, normalized database storage, SQL querying, and pandas analysis.**
