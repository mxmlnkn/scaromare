SPARK_ROOT=~/spark-1.5.2-bin-hadoop2.6
SPARKCORE_JAR=$(SPARK_ROOT)/lib/spark-assembly-1.5.2-hadoop2.6.0.jar
SCALA_ROOT=$$(dirname $$(which scala))/../lib
MAKE=make SPARK_ROOT=$(SPARK_ROOT) SPARKCORE_JAR=$(SPARKCORE_JAR) SCALA_ROOT=$(SCALA_ROOT)

all:
	$(MAKE) -C singleNode/singleCore/java
	$(MAKE) -C singleNode/singleCore/scala
	$(MAKE) -C singleNode/singleGpu/cpp
	$(MAKE) -C singleNode/singleGpu/java
	$(MAKE) -C singleNode/singleGpu/multiGpuTest
	$(MAKE) -C singleNode/multiGpu/java
	$(MAKE) -C singleNode/multiGpu/scala
	$(MAKE) -C multiNode/multiCore
	$(MAKE) -C multiNode/multiGpu/scala
