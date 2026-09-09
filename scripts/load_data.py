"""Load the Olist Brazilian E-Commerce dataset into a local DuckDB file.

Two paths, both producing the exact same schema in data/olist.duckdb:

1. Real data: if data/raw/*.csv already exist (downloaded manually from
   https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), load those.
2. Synthetic fallback: if no CSVs are present and KAGGLE credentials aren't
   configured, generate a small synthetic dataset with the identical schema
   so the whole app still runs fully offline.

Usage:
    python scripts/load_data.py            # auto: real if present, else synthetic
    python scripts/load_data.py --synthetic  # force synthetic
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "olist.duckdb"

# Maps the physical CSV filename (as distributed on Kaggle) to the simplified
# table name used throughout this project.
CSV_TO_TABLE = {
    "olist_customers_dataset.csv": "customers",
    "olist_orders_dataset.csv": "orders",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "payments",
    "olist_order_reviews_dataset.csv": "reviews",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "olist_geolocation_dataset.csv": "geolocation",
    "product_category_name_translation.csv": "category_translation",
}


def load_real_csvs(con: duckdb.DuckDBPyConnection) -> bool:
    if not RAW_DIR.exists():
        return False
    found = {f.name for f in RAW_DIR.glob("*.csv")}
    if not found.intersection(CSV_TO_TABLE):
        return False

    for csv_name, table in CSV_TO_TABLE.items():
        csv_path = RAW_DIR / csv_name
        if not csv_path.exists():
            print(f"  [skip] {csv_name} not found, table '{table}' will be empty/missing")
            continue
        con.execute(
            f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto(?, ignore_errors=true)",
            [str(csv_path)],
        )
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  [ok] {table}: {n:,} rows (from {csv_name})")
    return True


def generate_synthetic(con: duckdb.DuckDBPyConnection, seed: int = 42) -> None:
    """Generate a small synthetic dataset with the SAME schema/columns as the
    real Olist tables, so downstream SQL/tools/eval questions work unchanged.
    """
    random.seed(seed)
    n_customers = 500
    n_orders = 1200
    n_products = 150
    n_sellers = 60

    states = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "GO", "PE"]
    cities = {
        "SP": "sao paulo", "RJ": "rio de janeiro", "MG": "belo horizonte",
        "RS": "porto alegre", "PR": "curitiba", "SC": "florianopolis",
        "BA": "salvador", "DF": "brasilia", "GO": "goiania", "PE": "recife",
    }
    categories_pt_en = [
        ("cama_mesa_banho", "bed_bath_table"), ("beleza_saude", "health_beauty"),
        ("esporte_lazer", "sports_leisure"), ("moveis_decoracao", "furniture_decor"),
        ("informatica_acessorios", "computers_accessories"),
        ("utilidades_domesticas", "housewares"), ("brinquedos", "toys"),
        ("relogios_presentes", "watches_gifts"), ("telefonia", "telephony"),
        ("automotivo", "auto"),
    ]
    payment_types = ["credit_card", "boleto", "voucher", "debit_card"]
    order_statuses = ["delivered", "delivered", "delivered", "delivered", "shipped", "canceled", "processing"]

    start = datetime(2017, 1, 1)

    # category_translation
    con.execute("CREATE OR REPLACE TABLE category_translation (product_category_name VARCHAR, product_category_name_english VARCHAR)")
    con.executemany("INSERT INTO category_translation VALUES (?, ?)", categories_pt_en)

    # customers: customer_id is order-scoped, customer_unique_id is person-level (governed)
    con.execute("""
        CREATE OR REPLACE TABLE customers (
            customer_id VARCHAR, customer_unique_id VARCHAR,
            customer_zip_code_prefix VARCHAR, customer_city VARCHAR, customer_state VARCHAR
        )
    """)
    unique_people = [f"person_{i:05d}" for i in range(int(n_customers * 0.8))]
    customer_rows = []
    for i in range(n_customers):
        st = random.choice(states)
        customer_rows.append((
            f"cust_{i:05d}", random.choice(unique_people),
            f"{random.randint(10000,99999):05d}", cities[st], st,
        ))
    con.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customer_rows)

    # products
    con.execute("""
        CREATE OR REPLACE TABLE products (
            product_id VARCHAR, product_category_name VARCHAR,
            product_weight_g INTEGER, product_length_cm INTEGER,
            product_height_cm INTEGER, product_width_cm INTEGER
        )
    """)
    product_rows = [
        (f"prod_{i:05d}", random.choice(categories_pt_en)[0],
         random.randint(100, 5000), random.randint(5, 60),
         random.randint(5, 60), random.randint(5, 60))
        for i in range(n_products)
    ]
    con.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)", product_rows)

    # sellers
    con.execute("""
        CREATE OR REPLACE TABLE sellers (
            seller_id VARCHAR, seller_zip_code_prefix VARCHAR,
            seller_city VARCHAR, seller_state VARCHAR
        )
    """)
    seller_rows = []
    for i in range(n_sellers):
        st = random.choice(states)
        seller_rows.append((f"seller_{i:05d}", f"{random.randint(10000,99999):05d}", cities[st], st))
    con.executemany("INSERT INTO sellers VALUES (?, ?, ?, ?)", seller_rows)

    # geolocation
    con.execute("""
        CREATE OR REPLACE TABLE geolocation (
            geolocation_zip_code_prefix VARCHAR, geolocation_lat DOUBLE,
            geolocation_lng DOUBLE, geolocation_city VARCHAR, geolocation_state VARCHAR
        )
    """)
    geo_rows = []
    seen_zips = set(r[2] for r in customer_rows) | set(r[1] for r in seller_rows)
    for z in seen_zips:
        geo_rows.append((z, -25.0 + random.random() * 15, -50.0 + random.random() * 15,
                          random.choice(list(cities.values())), random.choice(states)))
    con.executemany("INSERT INTO geolocation VALUES (?, ?, ?, ?, ?)", geo_rows)

    # orders
    con.execute("""
        CREATE OR REPLACE TABLE orders (
            order_id VARCHAR, customer_id VARCHAR, order_status VARCHAR,
            order_purchase_timestamp TIMESTAMP, order_approved_at TIMESTAMP,
            order_delivered_carrier_date TIMESTAMP, order_delivered_customer_date TIMESTAMP,
            order_estimated_delivery_date TIMESTAMP
        )
    """)
    order_rows = []
    order_ids = []
    for i in range(n_orders):
        oid = f"order_{i:06d}"
        order_ids.append(oid)
        cust = random.choice(customer_rows)[0]
        status = random.choice(order_statuses)
        purchase = start + timedelta(days=random.randint(0, 700), hours=random.randint(0, 23))
        approved = purchase + timedelta(hours=random.randint(1, 48))
        carrier = approved + timedelta(days=random.randint(0, 3))
        est_delivery = purchase + timedelta(days=random.randint(7, 25))
        if status == "delivered":
            delivered = carrier + timedelta(days=random.randint(1, 20))
        else:
            delivered = None
        order_rows.append((oid, cust, status, purchase, approved, carrier, delivered, est_delivery))
    con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)", order_rows)

    # order_items
    con.execute("""
        CREATE OR REPLACE TABLE order_items (
            order_id VARCHAR, order_item_id INTEGER, product_id VARCHAR,
            seller_id VARCHAR, shipping_limit_date TIMESTAMP, price DOUBLE, freight_value DOUBLE
        )
    """)
    item_rows = []
    for oid in order_ids:
        n_items = random.choice([1, 1, 1, 2, 2, 3])
        for item_no in range(1, n_items + 1):
            price = round(random.uniform(15, 900), 2)
            freight = round(random.uniform(5, 60), 2)
            item_rows.append((
                oid, item_no, random.choice(product_rows)[0], random.choice(seller_rows)[0],
                start + timedelta(days=random.randint(0, 700)), price, freight,
            ))
    con.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?, ?)", item_rows)

    # payments
    con.execute("""
        CREATE OR REPLACE TABLE payments (
            order_id VARCHAR, payment_sequential INTEGER, payment_type VARCHAR,
            payment_installments INTEGER, payment_value DOUBLE
        )
    """)
    order_totals = {}
    for r in item_rows:
        order_totals[r[0]] = order_totals.get(r[0], 0) + r[5] + r[6]
    payment_rows = []
    for oid, total in order_totals.items():
        payment_rows.append((oid, 1, random.choice(payment_types), random.choice([1, 1, 2, 3, 6, 10]), round(total, 2)))
    con.executemany("INSERT INTO payments VALUES (?, ?, ?, ?, ?)", payment_rows)

    # reviews (only for delivered/shipped orders, some free-text comments to demo masking)
    con.execute("""
        CREATE OR REPLACE TABLE reviews (
            review_id VARCHAR, order_id VARCHAR, review_score INTEGER,
            review_comment_title VARCHAR, review_comment_message VARCHAR,
            review_creation_date TIMESTAMP, review_answer_timestamp TIMESTAMP
        )
    """)
    sample_comments = [
        "Produto chegou antes do prazo, recomendo!",
        "Nao gostei da qualidade, esperava mais.",
        "Entrega rapida, tudo certo.",
        "Produto veio com defeito.",
        None,
    ]
    review_rows = []
    for i, o in enumerate(order_rows):
        oid, status = o[0], o[2]
        if status not in ("delivered", "shipped"):
            continue
        if random.random() > 0.7:
            continue
        score = random.choices([1, 2, 3, 4, 5], weights=[5, 5, 10, 30, 50])[0]
        created = o[3] + timedelta(days=random.randint(3, 30))
        review_rows.append((
            f"rev_{i:06d}", oid, score, None, random.choice(sample_comments),
            created, created + timedelta(days=random.randint(0, 5)),
        ))
    con.executemany("INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?)", review_rows)

    print("  [ok] synthetic dataset generated:")
    for t in ["customers", "orders", "order_items", "payments", "reviews", "products", "sellers", "geolocation", "category_translation"]:
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"    {t}: {n:,} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data generation")
    args = parser.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    print(f"Loading data into {DB_PATH} ...")
    if args.synthetic:
        print("Forced synthetic mode.")
        generate_synthetic(con)
    else:
        loaded_real = load_real_csvs(con)
        if not loaded_real:
            print(f"No CSVs found in {RAW_DIR} -- falling back to synthetic dataset "
                  "(same schema, so the app runs fully offline).")
            print(f"To use the real dataset: download it from "
                  f"https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce, "
                  f"unzip the CSVs into {RAW_DIR}, then re-run this script.")
            generate_synthetic(con)

    con.close()
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
