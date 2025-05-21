import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import numpy as np


class PostgresDataIngestor:
    def __init__(self, host, port, database, user, password):
        self.conn = psycopg2.connect(
            host=host, port=port, database=database,
            user=user, password=password
        )
        self.cursor = self.conn.cursor()
        print("✅ Connected to PostgreSQL")

    def ingest_data(self, table_name, data_source, mode='append'):
        df = pd.read_csv(data_source) if isinstance(data_source, str) else data_source
        print(f"📦 Ingesting {len(df)} records into '{table_name}' (mode: {mode})")

        if df.empty:
            print("⚠️ No data to ingest.")
            return

        try:
            if mode == 'replace':
                self._create_table_if_not_exists(table_name, df)
                self._truncate_table(table_name)
            elif mode == 'append':
                self._create_table_if_not_exists(table_name, df)

            self._insert_dataframe(table_name, df)
            print(f"✅ Completed ingest for table '{table_name}'\n{'-'*50}")
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"❌ Error ingesting data into '{table_name}': {e}")

    def read_table(self, table_name):
        try:
            return pd.read_sql(f"SELECT * FROM {table_name}", self.conn)
        except Exception as e:
            print(f"❌ Error reading from table '{table_name}': {e}")
            return pd.DataFrame()

    def _create_table_if_not_exists(self, table_name, df):
        column_types = {
            'int': 'INTEGER',
            'float': 'FLOAT',
            'datetime': 'TIMESTAMP'
        }

        columns_sql = []
        for col, dtype in df.dtypes.items():
            for key, sql_type in column_types.items():
                if key in str(dtype):
                    break
            else:
                sql_type = 'TEXT'
            columns_sql.append(f'"{col}" {sql_type}')

        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(columns_sql)}
        )
        """
        self.cursor.execute(create_sql)
        self.conn.commit()
        print(f"✅ Table '{table_name}' verified/created.")

    def _truncate_table(self, table_name):
        self.cursor.execute(f"TRUNCATE TABLE {table_name}")
        self.conn.commit()
        print(f"🧹 Cleared data from '{table_name}'")

    def clear_table(self, table_name):
        try:
            self._truncate_table(table_name)
            print(f"✅ Cleared all data from '{table_name}'")
        except Exception as e:
            raise RuntimeError(f"❌ Error clearing data from '{table_name}': {e}")

    def _insert_dataframe(self, table_name, df):
        columns = ', '.join([f'"{col}"' for col in df.columns])
        placeholders = ', '.join(['%s'] * len(df.columns))
        insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        values = []
        for row in df.itertuples(index=False, name=None):
            clean_row = [
                None if pd.isna(v) else v.item() if isinstance(v, (np.generic,)) else v
                for v in row
            ]
            values.append(tuple(clean_row))

        execute_batch(self.cursor, insert_sql, values)
        self.conn.commit()
        print(f"📥 Inserted {len(df)} rows into '{table_name}'")

    def list_tables(self):
        try:
            self.cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
            tables = [row[0] for row in self.cursor.fetchall()]
            print(f"📋 Available tables: {tables}")
        except Exception as e:
            print(f"❌ Error listing tables: {e}")
            return []

    def close(self):
        try:
            self.cursor.close()
            self.conn.close()
            print("🔌 PostgreSQL connection closed")
        except Exception as e:
            print(f"❌ Error closing connection: {e}")

    def __del__(self):
        self.close()
