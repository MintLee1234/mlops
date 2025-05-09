from prometheus_client import start_http_server
import time

if __name__ == '__main__':
    start_http_server(8000)  # Port để Prometheus scrape metrics
    while True:
        time.sleep(10)