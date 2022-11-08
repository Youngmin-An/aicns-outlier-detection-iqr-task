"""
Outlier detectoion task using IQR method
"""

from func import *
from pyspark.sql import SparkSession


if __name__ == "__main__":
    # Initialize app
    app_conf = get_conf_from_evn()

    SparkSession.builder.config(
        "spark.hadoop.hive.exec.dynamic.partition", "true"
    ).config("spark.hadoop.hive.exec.dynamic.partition.mode", "nonstrict")

    # [AICNS-61]
    if app_conf["SPARK_EXTRA_CONF_PATH"] != "":
        config_dict = parse_spark_extra_conf(app_conf)
        for conf in config_dict.items():
            SparkSession.builder.config(conf[0], conf[1])

    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel(app_conf["APP_LOG_LEVEL"])

    # Get feature metadata
    data_col_name = (
        "input_data"  # todo: metadata concern or strict validation column names
    )
    time_col_name = "event_time"

    # Load sample data
    ts = load_sample_data(app_conf, time_col_name, data_col_name)

    # Fetch inferential stats(q1, median, q3)
    q1, median, q3 = fetch_stats(app_conf)

    # Detect anomalies
    detected_df = detect_anomalies(
        ts, time_col_name, data_col_name, q1, median, q3, app_conf
    )

    # Save target data
    save_target_data_to_dwh(detected_df, app_conf, time_col_name, data_col_name)

    spark.stop()
