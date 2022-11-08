"""
Function level adapters
"""
import os
import pendulum
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from univariate.anomaly.detector import IQROutlierDetector
from univariate.analyzer import AnalysisReport
from typing import Tuple


__all__ = [
    "get_conf_from_evn",
    "parse_spark_extra_conf",
    "load_sample_data",
    "fetch_stats",
    "detect_anomalies",
    "save_target_data_to_dwh",
]

# todo: singleton logger in adaptor class
# logger = SparkSession.getActiveSession()._jvm.org.apache.log4j.LogManager.getLogger(__name__)


def get_conf_from_evn():
    """
        Get conn info from env variables
    :return:
    """
    conf = dict()
    try:
        # todo: enable configure ad-hoc detection mode
        # Feature id
        conf["FEATURE_ID"] = os.getenv("FEATURE_ID")
        # Raw data period
        sample_start_datetime = os.getenv(
            "APP_DATA_SAMPLE_START"
        )  # yyyy-MM-dd'T'HH:mm:ss
        sample_end_datetime = os.getenv("APP_DATA_SAMPLE_END")  # yyyy-MM-dd'T'HH:mm:ss
        target_start_datetime = os.getenv(
            "APP_ANALYSIS_TARGET_START"
        )  # yyyy-MM-dd'T'HH:mm:ss
        target_end_datetime = os.getenv(
            "APP_ANALYSIS_TARGET_END"
        )  # yyyy-MM-dd'T'HH:mm:ss
        conf["APP_TIMEZONE"] = os.getenv("APP_TIMEZONE", default="UTC")

        conf["SPARK_EXTRA_CONF_PATH"] = os.getenv(
            "SPARK_EXTRA_CONF_PATH", default=""
        )  # [AICNS-61]
        conf["SAMPLE_START"] = pendulum.parse(sample_start_datetime).in_timezone(
            conf["APP_TIMEZONE"]
        )
        conf["SAMPLE_END"] = pendulum.parse(sample_end_datetime).in_timezone(
            conf["APP_TIMEZONE"]
        )
        conf["TARGET_START"] = pendulum.parse(target_start_datetime).in_timezone(
            conf["APP_TIMEZONE"]
        )
        conf["TARGET_END"] = pendulum.parse(target_end_datetime).in_timezone(
            conf["APP_TIMEZONE"]
        )

        conf["APP_LOG_LEVEL"] = os.getenv("APP_LOG_LEVEL", default="WARN")
        # todo: temp patch for day resolution parsing, so later with [AICNS-59] resolution will be subdivided.
        # conf["SAMPLE_END"] = conf["SAMPLE_END"].subtract(minutes=1)
        # conf["TARGET_END"] = conf["TARGET_END"].subtract(minutes=1)

    except Exception as e:
        print(e)
        raise e
    return conf


def parse_spark_extra_conf(app_conf):
    """
    Parse spark-default.xml style config file.
    It is for [AICNS-61] that is spark operator take only spark k/v confs issue.
    :param app_conf:
    :return: Dict (key: conf key, value: conf value)
    """
    with open(app_conf["SPARK_EXTRA_CONF_PATH"], "r") as cf:
        lines = cf.read().splitlines()
        config_dict = dict(
            list(
                filter(
                    lambda splited: len(splited) == 2,
                    (map(lambda line: line.split(), lines)),
                )
            )
        )
    return config_dict


def load_sample_data(app_conf, time_col_name, data_col_name) -> DataFrame:
    logger = SparkSession.getActiveSession()._jvm.org.apache.log4j.LogManager.getLogger(
        __name__
    )
    table_name = "cleaned_deduplicated"
    # Inconsistent cache
    # https://stackoverflow.com/questions/63731085/you-can-explicitly-invalidate-the-cache-in-spark-by-running-refresh-table-table
    SparkSession.getActiveSession().sql(f"REFRESH TABLE {table_name}")
    query = f"""
    SELECT v.{time_col_name}, v.{data_col_name}  
        FROM (
            SELECT {time_col_name}, {data_col_name}, concat(concat(cast(year as string), lpad(cast(month as string), 2, '0')), lpad(cast(day as string), 2, '0')) as date 
            FROM {table_name}
            WHERE feature_id = {app_conf["FEATURE_ID"]}
            ) v 
        WHERE v.date  >= {app_conf['SAMPLE_START'].format('YYYYMMDD')} AND v.date < {app_conf['SAMPLE_END'].format('YYYYMMDD')} 
    """
    logger.debug("load sample data query: " + query)
    ts = SparkSession.getActiveSession().sql(query)
    logger.debug("sample data")
    ts.show()
    return ts.sort(F.col(time_col_name).desc())


def __temp_fetch(app_conf):
    """
    Fetch from last descriptive stats by sample
    :param app_conf:
    :return:
    """
    logger = SparkSession.getActiveSession()._jvm.org.apache.log4j.LogManager.getLogger(
        __name__
    )
    stats = (
        SparkSession.getActiveSession()
        .sql(
            f"SELECT Q1, median, Q3 FROM descriptive_stat_experiments WHERE feature_id={app_conf['FEATURE_ID']} ORDER BY sample_end LIMIT 1"
        )
        .first()
        .asDict()
    )
    logger.debug(f'Q1: {stats["Q1"]}, median: {stats["median"]}, Q3: {stats["Q3"]}')
    return stats["Q1"], stats["median"], stats["Q3"]


def fetch_stats(app_conf) -> Tuple[float, float, float]:
    """

    :param app_conf:
    :return: (Q1, median, Q3) about feature_id
    """
    # todo: from inferential stats
    logger = SparkSession.getActiveSession()._jvm.org.apache.log4j.LogManager.getLogger(
        __name__
    )
    q1, median, q3 = __temp_fetch(app_conf)
    logger.debug(f"Q1: {q1}, median: {median}, Q3: {q3}")
    return q1, median, q3


def detect_anomalies(
    ts: DataFrame,
    time_col_name: str,
    data_col_name: str,
    q1: float,
    median: float,
    q3: float,
    app_conf,
) -> DataFrame:
    """

    :param ts:
    :param time_col_name:
    :param data_col_name:
    :param q1:
    :param median:
    :param q3:
    :param app_conf:
    :return:
    """
    logger = SparkSession.getActiveSession()._jvm.org.apache.log4j.LogManager.getLogger(
        __name__
    )
    report: AnalysisReport = IQROutlierDetector.detect(
        ts, data_col_name, q1, median, q3
    )
    df: DataFrame = report.parameters["outliers"]  # todo: rename parameters key
    target_start = app_conf["TARGET_START"].timestamp() * 1000
    target_end = app_conf["TARGET_END"].timestamp() * 1000
    logger.debug(f"target_start: {target_start}, target_end: {target_end}")
    target_df = df.filter(
        (F.col(time_col_name) >= target_start) & (F.col(time_col_name) < target_end)
    )
    logger.debug(f"target filtered df count: {target_df.count()}")
    return target_df  # todo: who has responsibility of parsing report (client? to propagate report)


def __append_partition_cols(
    ts: DataFrame, time_col_name: str, data_col_name: str, feature_id: str
):
    return (
        ts.withColumn("datetime", F.from_unixtime(F.col(time_col_name) / 1000))
        .select(
            time_col_name,
            data_col_name,
            "is_anomaly_by_iqr_method",
            F.lit(feature_id).alias("feature_id"),
            F.year("datetime").alias("year"),
            F.month("datetime").alias("month"),
            F.dayofmonth("datetime").alias("day"),
        )
        .sort(time_col_name)
    )


def save_target_data_to_dwh(
    ts: DataFrame, app_conf, time_col_name: str, data_col_name: str
):

    # todo: transaction
    logger = SparkSession.getActiveSession()._jvm.org.apache.log4j.LogManager.getLogger(
        __name__
    )
    table_name = "cleaned_detected_anomalies_iqr"
    query = f"CREATE TABLE IF NOT EXISTS {table_name} ({time_col_name} BIGINT, {data_col_name} DOUBLE, is_anomaly_by_iqr_method BOOLEAN) PARTITIONED BY (feature_id CHAR(10), year int, month int, day int) STORED AS PARQUET"
    logger.debug(f"creating table query: {query}")
    SparkSession.getActiveSession().sql(query)

    period = pendulum.period(app_conf["TARGET_START"], app_conf["TARGET_END"])

    # Create partition columns(year, month, day) from timestamp
    partition_df = __append_partition_cols(
        ts, time_col_name, data_col_name, app_conf["FEATURE_ID"]
    )

    for date in period.range("days"):
        # Drop Partition for immutable task
        query = f"ALTER TABLE {table_name} DROP IF EXISTS PARTITION(feature_id={app_conf['FEATURE_ID']}, year={date.year}, month={date.month}, day={date.day})"
        logger.debug(f"Partition check query: {query}")
        SparkSession.getActiveSession().sql(query)
    # Save
    partition_df.write.format("hive").mode("append").insertInto(table_name)
