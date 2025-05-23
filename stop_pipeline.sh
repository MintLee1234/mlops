#!/bin/bash

echo "🛑 Đang dừng toàn bộ hệ thống: Airflow, Prometheus, StatsD Exporter..."

# === Dừng Airflow Webserver ===
WEB_PID=$(ps aux | grep 'airflow webserver' | grep -v grep | awk '{print $2}')
if [ -n "$WEB_PID" ]; then
    kill "$WEB_PID"
    echo "✅ Đã dừng Airflow Webserver (PID: $WEB_PID)"
else
    echo "⚠️ Không tìm thấy Airflow Webserver đang chạy."
fi

# === Dừng Airflow Scheduler ===
SCHED_PID=$(ps aux | grep 'airflow scheduler' | grep -v grep | awk '{print $2}')
if [ -n "$SCHED_PID" ]; then
    kill "$SCHED_PID"
    echo "✅ Đã dừng Airflow Scheduler (PID: $SCHED_PID)"
else
    echo "⚠️ Không tìm thấy Airflow Scheduler đang chạy."
fi

# === Dừng Prometheus ===
PROM_PID=$(ps aux | grep 'prometheus.*--config.file' | grep -v grep | awk '{print $2}')
if [ -n "$PROM_PID" ]; then
    kill "$PROM_PID"
    echo "✅ Đã dừng Prometheus (PID: $PROM_PID)"
else
    echo "⚠️ Không tìm thấy Prometheus đang chạy."
fi

# === Dừng StatsD Exporter ===
STATSD_PID=$(ps aux | grep 'statsd_exporter' | grep -v grep | awk '{print $2}')
if [ -n "$STATSD_PID" ]; then
    kill "$STATSD_PID"
    echo "✅ Đã dừng StatsD Exporter (PID: $STATSD_PID)"
else
    echo "⚠️ Không tìm thấy StatsD Exporter đang chạy."
fi

echo "🛑 Toàn bộ thành phần đã được dừng."
