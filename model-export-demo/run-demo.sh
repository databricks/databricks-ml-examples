#!/usr/bin/env bash

# This script runs the demo after deleting dbml-local from the local .m2 cache.
# This forces mvn to download the artifacts from bintray, in order to QA the public
# release artifacts.
#
# Usage:
#  Run from the dbml-local-demo directory.
#  ./qa-demo.sh

set -euo pipefail

rm -rf ~/.m2/repository/com/databricks/dbml-local/

mvn clean
mvn compile
mvn exec:java@app-vector-input
mvn exec:java@app-string-input
mvn exec:java@app-multi-threading
