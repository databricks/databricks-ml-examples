import IPython

import logging

logger = logging.getLogger(__name__)


def get_dbutils():
    """
    Returns dbutils object. Works only on Databricks.
    """
    if IPython.get_ipython() is not None:
        return IPython.get_ipython().user_ns["dbutils"]
    else:
        raise RuntimeError(
            f"Could not retrieve dbutils because not running in a Databricks notebook!"
        )
