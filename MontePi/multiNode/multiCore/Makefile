ROOTBEER_ROOT := ../../../rootbeer1
SPARK_VERSION := 1.5.2
SPARK_ROOT    := /opt/spark-$(SPARK_VERSION)
SCALA_ROOT    := /usr/share/java

SPARKCORE_JAR := $(SPARK_ROOT)/core/target/spark-core_2.10-$(SPARK_VERSION).jar
HADOOP_JAR    := $(SPARK_ROOT)/assembly/target/scala-2.10/spark-assembly-$(SPARK_VERSION)-hadoop2.2.0.jar

.phony: clean all *.jar

all: MontePi.jar
run: MontePi.jar
	$(SPARK_ROOT)/bin/spark-submit --master local[4] --class TestMonteCarloPi $< 268435456 4
clean:
	rm -f *.class *.jar

MonteCarloPiKernel.class: MonteCarloPiKernel.java
	javac $< -classpath .
MonteCarloPi.class: MonteCarloPi.java MonteCarloPiKernel.class
	javac $< -classpath .
TestMonteCarloPi.class: TestMonteCarloPi.scala MonteCarloPi.class
	scalac -classpath $(SPARKCORE_JAR):$(HADOOP_JAR):. -deprecation $<

MontePi.jar: TestMonteCarloPi.class manifest.txt
	jar -cvfm $@.tmp.jar manifest.txt *.class
	zipmerge $@ $(SCALA_ROOT)/scala-library.jar $@.tmp.jar
