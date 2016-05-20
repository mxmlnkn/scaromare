SPARK_ROOT=~/spark-1.5.2-bin-hadoop2.6
SPARKCORE_JAR=$(SPARK_ROOT)/lib/spark-assembly-1.5.2-hadoop2.6.0.jar
SCALA_ROOT=$$(dirname $$(which scala))/../lib

all:
	make -C multiNode/multiCore SPARK_ROOT=$(SPARK_ROOT) SPARKCORE_JAR=$(SPARKCORE_JAR) SCALA_ROOT=$(SCALA_ROOT)
	make -C multiNode/multiCore SPARK_ROOT=$(SPARK_ROOT) SPARKCORE_JAR=$(SPARKCORE_JAR) SCALA_ROOT=$(SCALA_ROOT)
