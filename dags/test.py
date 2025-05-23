import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monitoring.metric_server import prediction_class_0, prediction_class_1, daily_crawled_count

print(prediction_class_0)
print(prediction_class_1)
print(daily_crawled_count)
