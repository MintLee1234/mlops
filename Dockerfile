# Sử dụng image chính thức của Airflow
FROM apache/airflow:2.8.1-python3.10

USER root

# (Tùy chọn) Cài thêm thư viện hệ thống nếu cần
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean

USER airflow

ENV PATH="/home/airflow/.local/bin:$PATH"

# Sao chép file requirements.txt vào container
COPY requirements.txt /requirements.txt

# Cài đặt thư viện Python
RUN pip install --no-cache-dir -r /requirements.txt
