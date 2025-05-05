import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import joblib
from category_encoders import TargetEncoder

class DataTransformation:
    def __init__(self):
        pass

    def transform_data(self, df):
        """
        Performs data transformations on the input DataFrame.
        """

        df = df.copy()  # tránh SettingWithCopyWarning

        user_id = df['user_id']
        df.drop(columns=['user_id'], inplace=True)

        # Convert data types
        numeric_cols = ['age', 'days_since_last_login', 'avg_time_spent', 'avg_transaction_value',
                        'avg_frequency_login_days', 'points_in_wallet']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df['last_visit_time'] = pd.to_datetime(df['last_visit_time'], format='%H:%M:%S', errors='coerce')

        categorical_cols = ['gender', 'region_category', 'membership_category',
                            'joined_through_referral', 'preferred_offer_types', 'medium_of_operation',
                            'internet_option', 'used_special_discount', 'offer_application_preference',
                            'past_complaint', 'complaint_status', 'feedback']
        df[categorical_cols] = df[categorical_cols].astype('object')
        df['joining_date'] = pd.to_datetime(df['joining_date'], errors='coerce')

        # Handle missing values
        df['gender'] = df['gender'].replace('Unknown', np.nan)

        # Impute numerical columns
        scaler = StandardScaler()
        imputer_num = SimpleImputer()
        df_scaled = df[numeric_cols].copy()
        df_scaled = scaler.fit_transform(df_scaled)
        df_scaled = imputer_num.fit_transform(df_scaled)
        df[numeric_cols] = scaler.inverse_transform(df_scaled)

        # Impute categorical columns
        cat_impute_cols = ['gender', 'region_category', 'joined_through_referral',
                        'medium_of_operation', 'preferred_offer_types']
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_cat = encoder.fit_transform(df[cat_impute_cols])
        imputer_cat = KNNImputer(n_neighbors=5, weights='distance', metric='nan_euclidean')
        df_cat_imputed = imputer_cat.fit_transform(df_cat)
        df[cat_impute_cols] = encoder.inverse_transform(df_cat_imputed).astype('object')

        # Feature engineering
        today = datetime(2024, 5, 17)  # hoặc datetime.today() nếu muốn tính động
        df['tenure_months'] = ((today.year - df['joining_date'].dt.year) * 12 +
                            (today.month - df['joining_date'].dt.month)).astype('int64')
        df['visit_hour'] = df['last_visit_time'].dt.hour.astype('Int64')

        df['login_spend_ratio'] = np.where(df['avg_frequency_login_days'] == 0, 0,
                                        df['avg_time_spent'] / df['avg_frequency_login_days'])
        df['login_transaction_ratio'] = np.where(df['avg_transaction_value'] == 0, 0,
                                                df['avg_frequency_login_days'] / df['avg_transaction_value'])

        df.drop(columns=['last_visit_time'], inplace=True)

        # Rename columns
        df.rename(columns={
            'avg_frequency_login_days': 'frequency',
            'avg_transaction_value': 'monetary',
            'days_since_last_login': 'recency'
        }, inplace=True)

        df['user_id'] = user_id
        return df
    
    def get_transformer_obj(self, X_train, y_train):
        numerical_cols = [
            'age', 'recency', 'avg_time_spent', 'monetary', 'frequency',
            'points_in_wallet', 'tenure_months', 'visit_hour',
            'login_spend_ratio', 'login_transaction_ratio'
        ]
        
        categorical_cols = [
            'gender','region_category', 'membership_category', 'joined_through_referral',
            'preferred_offer_types', 'internet_option', 'used_special_discount',
            'offer_application_preference', 'past_complaint', 'complaint_status',
            'feedback', 'medium_of_operation'
        ]

        # Pipelines for each type of feature
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('target_encoder', TargetEncoder(cols=categorical_cols))
        ])

        # Create ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('numerical', numeric_transformer, numerical_cols),
                ('categorical', categorical_transformer, categorical_cols)
            ],
            remainder='passthrough'  # Keep any extra columns as-is (e.g. 'user_id')
        )

        # Fit ONLY — do NOT transform here
        preprocessor.fit(X_train, y_train)

        # Save the preprocessor
        os.makedirs('preprocessors', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'preprocessor_{timestamp}.pkl'
        filepath = os.path.join('preprocessors', filename)

        joblib.dump(preprocessor, filepath)

        # Log the save time and path
        with open('preprocessors/preprocessor_versions.txt', 'a') as log_file:
            log_file.write(f'{datetime.now()} - {filepath}\n')

        return preprocessor

    def train_val_test_splitting(self, df):
            self.df = df
            X = self.df.drop(columns=["churn_risk_score"])
            y = self.df["churn_risk_score"]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)  
            return self.X_train, self.X_test, self.y_train, self.y_test

        
    def initiate_data_transformation(self, df):
            self.df = df
            self.X_train, self.X_test, self.y_train, self.y_test = self.train_val_test_splitting(self.df)
            preprocessor_obj = self.get_transformer_obj(self.X_train, self.y_train)

            self.X_train_transformed = preprocessor_obj.transform(self.X_train)
            self.X_test_transformed = preprocessor_obj.transform(self.X_test)

            pd.DataFrame(self.X_train_transformed).to_csv('/home/minhle/mlops/data/X_train_transformed.csv', index=False)
            pd.DataFrame(self.X_test_transformed).to_csv('/home/minhle/mlops/data/X_test_transformed.csv', index=False)
            pd.DataFrame(self.y_train).to_csv('/home/minhle/mlops/data/y_train.csv', index=False)
            pd.DataFrame(self.y_test).to_csv('/home/minhle/mlops/data/y_test.csv', index=False)
