from statsd import StatsClient

statsd = StatsClient(host='localhost', port=9125, prefix='airflow')


class Metric:
    def __init__(self, name):
        self.name = name

    def set(self, value):
        statsd.gauge(self.name, value)


prediction_class_0 = Metric('prediction_class_0')
prediction_class_1 = Metric('prediction_class_1')
daily_crawled_count = Metric('daily_crawled_count')
