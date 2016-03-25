@Patrick - if we make a habit/convention of hosting the data as a sibling of the project root then we can run the code without fixing pathing and without committing data to the repo.  e.g.

```python
print('Load data...')
DATA_DIR = "../../bnp-data"
train = pd.read_csv(DATA_DIR + "/train.csv")
test = pd.read_csv(DATA_DIR + "/test.csv")
```

== Spark Considerations
Databricks the guys who really ARE spark offer this sklearn/spark integration which I'm trying to take advantage of.  https://github.com/databricks/spark-sklearn

There's another project I have not played with that gives some custom types of RDD's and have another approach for integration sklearn with spark. https://github.com/lensacom/sparkit-learn



== Local Stuff

=== Getting Up And Running

To run the code for "Optimization and Cross Validation.ipynb" in a Jupyter notebook, aside from installation of Anaconda/Conda I created a custom environment paribas-spark and also had to

```bash
pip install xgboost
pip install -U scikit-learn
pip install scikit-neuralnetwork
pip install tabulate
pip install spark-sklearn


```

=== Speed
For faster iterations on just the dev side I created some _1k versions of the files and use those when I want like 30 second runs for quick iterations.

=== Intellij Notes

Environment vars in config
PYTHONPATH /Users/cmathias/chris/data-dev/spark/python/:/Users/cmathias/chris/data-dev/spark/python/lib/py4j-0.9-src.zip:$PYTHONPATH

SPARK_HOME
/Users/cmathias/chris/data-dev/spark/

=== Spark Notes

Created spark-env.sh from spark-env-template.sh and added
SPARK_DRIVER_MEMORY=2G
SPARK_CLASSPATH=/Users/cmathias/chris/data-dev/spark/lib-ext/spark-csv_2.10-1.3.0.jar:/Users/cmathias/chris/data-dev/spark/lib-ext/commons-csv-1.1.jar


== Cluster Stuff
=== Databricks Notebook Setup Notes

https://forums.databricks.com/questions/680/how-to-install-python-package-on-spark-cluster.html

Had to manually add the packages indicated above in 'getting up and running' via the mechanism described above.  Each should now be automatically applied when the cluster is brought up.  This is NOT a quick operation however we're paying by the hour so it should be brought down when not in use.  Give it 5 - 10 minutes to come up clean.

This didn't actually work for xgboost. For xgboost I had to use a "manual" pip install, e.g. within the notebook itself:

```bash
%sh /home/ubuntu/databricks/python/bin/pip install xgboost --pre
```
This is an undocumented feature uncovered by complaining to the service about the fact that I wasn't able to get XGBoost going.  https://forums.databricks.com/questions/7441/cannot-successfully-install-xgboost-on-databricks.html

Notebook shared to Patrick: https://dbc-e13e992a-67d9.cloud.databricks.com/#notebook/3921









