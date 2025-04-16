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

from category_encoders import TargetEncoder

class DataTransformation:
    def __init__(self, df):
        self.df = self.transform_data(df)
    def transform_data(self, df):
        """
        Performs data transformations on the input DataFrame.
        """

        # Convert data to appropriate dtypes
        numerical_columns = ['age', 'days_since_last_login', 'avg_time_spent',
                            'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet', 'churn_risk_score']
        df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')
        df['last_visit_time'] = pd.to_datetime(df['last_visit_time'], format='%H:%M:%S')
        categorical_columns = ['gender', 'region_category', 'membership_category',
                            'joined_through_referral', 'preferred_offer_types', 'medium_of_operation',
                            'internet_option', 'used_special_discount', 'offer_application_preference',
                            'past_complaint', 'complaint_status', 'feedback']
        df[categorical_columns] = df[categorical_columns].astype('object')
        df['joining_date'] = pd.to_datetime(df['joining_date'])

        # Impute missing values
        # Iterative Imputer for numerical columns
        target_column = 'churn_risk_score'
        numeric_columns = df.select_dtypes(include='number').columns.drop(target_column)
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        imputer = SimpleImputer()
        df_scaled[numeric_columns] = imputer.fit_transform(df_scaled[numeric_columns])
        df[numeric_columns] = scaler.inverse_transform(df_scaled[numeric_columns])

        # KNN Imputer for categorical columns
        df['gender'] = df['gender'].replace('Unknown', np.nan)
        categorical_columns = ['gender', 'region_category', 'joined_through_referral', 'medium_of_operation',
                            'preferred_offer_types']
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoder.fit(df[categorical_columns])
        df[categorical_columns] = encoder.transform(df[categorical_columns])
        imputer = KNNImputer(n_neighbors=5, metric='nan_euclidean', weights='distance')
        df[categorical_columns] = imputer.fit_transform(df[categorical_columns])
        for col in categorical_columns:
            df[col] = df[col].astype('object')

        # Feature Engineering
        specific_date = datetime(2024, 5, 17)
        df['tenure_months'] = ((specific_date.year - df['joining_date'].dt.year) * 12 +
                            (specific_date.month - df['joining_date'].dt.month)).astype('int64')
        df['visit_hour'] = df['last_visit_time'].dt.hour.astype('int64')
        df['login_spend_ratio'] = df.apply(
            lambda row: 0 if row['avg_frequency_login_days'] == 0 else row['avg_time_spent'] / row['avg_frequency_login_days'],
            axis=1
        )
        df['login_transaction_ratio'] = df.apply(
            lambda row: 0 if row['avg_transaction_value'] == 0 else row['avg_frequency_login_days'] / row['avg_transaction_value'],
            axis=1
        )

        df = df.drop(columns=['joining_date', 'last_visit_time'])

        # Rename columns
        rename_mapping = {
            'avg_frequency_login_days': 'frequency',
            'avg_transaction_value': 'monetary',
            'days_since_last_login': 'recency'
        }

        df = df.rename(columns=rename_mapping)
        print(df.info())

        return df
    def get_transformer_obj(self, X_train, y_train):
            numerical_cols = ['age',
                            'recency',
                            'avg_time_spent',
                            'monetary',
                            'frequency',
                            'points_in_wallet',
                            'tenure_months',
                            'visit_hour',
                            'login_spend_ratio',
                            'login_transaction_ratio'
                        ]
            categorical_cols = ['gender',
                                'region_category',
                                'membership_category',
                                'joined_through_referral',
                                'preferred_offer_types',
                                'internet_option',
                                'used_special_discount',
                                'offer_application_preference',
                                'past_complaint',
                                'complaint_status',
                                'feedback',
                                'medium_of_operation'
                            ]

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('target_encoder', TargetEncoder(cols=categorical_cols))

            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical', numeric_transformer, numerical_cols),
                    ('categorical', categorical_transformer, categorical_cols)
                ],
                remainder='passthrough'
            )

            preprocessor.fit(X_train, y_train)

            return preprocessor


    def train_val_test_splitting(self):
            X = self.df.drop(columns=["churn_risk_score"])
            y = self.df["churn_risk_score"]
            X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% train, 30% remaining
            X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42) # Split remaining 30% 


            return X_train, X_val, X_test, y_train, y_val, y_test

    def initiate_data_transformation(self, X_train, X_val, X_test, y_train, y_val, y_test):

            preprocessor_obj = self.get_transformer_obj(X_train, y_train)

            X_train_transformed = preprocessor_obj.transform(X_train)
            X_val_transformed = preprocessor_obj.transform(X_val)
            X_test_transformed = preprocessor_obj.transform(X_test)

            return X_train_transformed, X_val_transformed, X_test_transformed, y_train, y_val, y_test

