#!/bin/bash

# === 1. Đặt biến môi trường ===
export AIRFLOW_HOME=/home/minhle/mlops

echo "🚀 Đang khởi động các thành phần của hệ thống..."

# === 2. Khởi động Airflow ===
echo "🌀 Airflow Home: $AIRFLOW_HOME"
airflow webserver -p 8080 > /tmp/airflow_webserver.log 2>&1 &
airflow scheduler > /tmp/airflow_scheduler.log 2>&1 &
echo "✅ Airflow đang chạy tại: http://localhost:8080"

# === 3. Khởi động Prometheus ===
PROMETHEUS_DIR="/home/minhle/mlops/prometheus/prometheus-3.4.0.linux-amd64"
$PROMETHEUS_DIR/prometheus --config.file=$PROMETHEUS_DIR/prometheus.yml > /tmp/prometheus.log 2>&1 &
echo "✅ Prometheus chạy tại: http://localhost:9090"

# === 4. Khởi động StatsD Exporter ===
STATSD_DIR="/home/minhle/mlops/prometheus/statsd_exporter-0.28.0.linux-amd64"
$STATSD_DIR/statsd_exporter > /tmp/statsd_exporter.log 2>&1 &
echo "✅ StatsD Exporter chạy tại: http://localhost:9102/metrics"

echo "✅ Tất cả dịch vụ đã được khởi động thành công!"
