import pymongo
import pandas as pd
import os
import dotenv
import logging
import time
from datetime import datetime

class DataIngestion:
    def __init__(self, mongo_uri, db_name, collection_name):
        self.mongo_uri = mongo_uri
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db_name = db_name
        self.collection_name = collection_name
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

    def upload_csv_to_mongodb(self, csv_file_path):
        
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} records from {csv_file_path}")
        
        data = df.to_dict(orient='records')
        if data:
            self.collection.insert_many(data)
            print(f"Inserted {len(data)} records into {self.db_name}.{self.collection_name}")
        else:
            print("No data to insert.")
        

    def download_data_from_mongodb(self):
        # Lấy toàn bộ documents
        data = list(self.collection.find())

        if not data:
            print("Không có dữ liệu trong collection.")
            return

        # Xoá cột `_id` của MongoDB (tuỳ bạn giữ hay xoá)
        for item in data:
            item.pop('_id', None)

        # Chuyển sang DataFrame
        df = pd.DataFrame(data)
        return df
    
    def delete_all_data(self):
        result = self.collection.delete_many({})
        print(f"Đã xoá {result.deleted_count} documents khỏi {self.db_name}.{self.collection_name}")


        # Đóng kết nối
    def __del__(self):
        self.client.close()
        print("MongoDB connection closed.")

