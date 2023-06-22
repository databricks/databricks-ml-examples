# [DEPRECATED] Demo application for Databricks Model Export

> **Warning**
Databricks ML Model Export is deprecated and removed from Databricks Runtime 6.0 and above.


This repository contains demos showing how to export Apache Spark ML models using [Databricks ML Model Export](https://docs.databricks.com/spark/latest/mllib/index.html)
and use them in your own application.

- The notebooks under `./notebooks` shows how to export an MLlib model on Databricks. They can [be uploaded to Databricks](https://docs.databricks.com/user-guide/notebooks/index.html#importing-notebooks).
- The Java demo apps under `./src` show how to import the model and perform inferernce via `dbml-local`.

For details including *supported models* and *versioning*, please check out [Databricks Documentation on Machine Learning](https://docs.databricks.com/spark/latest/mllib/index.html).

### Instructions to run

Use the accompanying notebook to train and export MLlib models. The `my_models/` directory also
contains example models you can use to get started.

Make sure that you have Maven installed, and then run:
```bash
mvn compile
mvn exec:java@app-vector-input
mvn exec:java@app-string-input
mvn exec:java@app-multi-threading
```
The calls to Maven exec will execute the example applications.

### `dbml-local` package

See the `pom.xml` file for the one dependency required by Databricks ML model export: `dbml-local`.
This package is published at [https://dl.bintray.com/databricks/maven/com/databricks/dbml-local/].
