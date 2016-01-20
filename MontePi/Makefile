SCALA_FILES   := $(wildcard *.scala)
CLASS_FILES   := $(SCALA_FILES:.scala=.class)
ROOTBEER_ROOT := ../rootbeer1
SPARK_ROOT    := /opt/spark-1.5.2
SCALA_ROOT    := /usr/share/java

.phony: clean all *.class *.jar

all: MontePiScala.jar MontePi_SparkRootbeer.jar

clean:
	$(MAKE) -C gpu clean
	rm -f *.class *.jar


MontePi_SparkRootbeer.jar: TestMonteCarloPi_SparkRootbeer.jar
	$(MAKE) -C gpu
	zipmerge $@ $(SPARK_ROOT)/core/target/spark-core_2.10-1.5.2.jar gpu/MontePiGPU.jar $<
	$(SPARK_ROOT)/bin/spark-submit --class TestMonteCarloPi_SparkRootbeer $@

TestMonteCarloPi_SparkRootbeer.jar: TestMonteCarloPi_SparkRootbeer.class
	jar cfm $@ manifest_SparkRootbeer.txt $*

TestMonteCarloPi_SparkRootbeer.class: TestMonteCarloPi_SparkRootbeer.java
	javac -classpath $(SPARK_ROOT)/core/target/spark-core_2.10-1.5.2.jar:gpu/MontePiGPU.jar:. $<



MontePiScala.jar: TestMonteCarloPi.jar
	$(MAKE) -C gpu
	# merge dependencies (Rootbeer,Scala) into a fat jar
	# Note that the rightmost jar overwrites all to the left, this is
	# important for the manifest, because we want to use the manifest
	# of TestMonteCarloPi.jar
	zipmerge $@ gpu/MontePiGPU.jar $(SCALA_ROOT)/scala-library.jar $<
	java -jar $@

TestMonteCarloPi.jar: TestMonteCarloPi.class
	jar cfm $@ manifest.txt $*

TestMonteCarloPi.class: TestMonteCarloPi.scala
	scalac -classpath gpu/MontePiGPU.jar:. $< -deprecation
