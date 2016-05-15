ROOTBEER_ROOT := ../../../../rootbeer1
SPARK_ROOT    := /opt/spark-1.5.2
SCALA_ROOT    := /usr/share/java
SCALAC        := scalac
JAVAC         := javac

SPARKCORE_JAR = $(SPARK_ROOT)/core/target/spark-core_2.10-1.5.2.jar
HADOOP_JAR    = $(SPARK_ROOT)/assembly/target/scala-2.10/spark-assembly-1.5.2-hadoop2.2.0.jar

.phony: clean all *.jar *.class

all: MontePi.jar
run: MontePi.jar
	$(SPARK_ROOT)/bin/spark-submit --master local[4] --class TestMonteCarloPi $< 268435456 1
clean:
	$(MAKE) -C gpu clean
	rm -f *.class *.jar

TestMonteCarloPi.class: TestMonteCarloPi.scala
	$(MAKE) -C gpu MontePiGPU.jar
	$(SCALAC) $< -classpath $(HADOOP_JAR):$(SPARKCORE_JAR):gpu/MontePiGPU.jar:. -deprecation

TestMontePi.jar: TestMonteCarloPi.class
	jar -cvfm $@ manifest.txt *.class

MontePi.jar: TestMontePi.jar
	$(MAKE) -C gpu MontePiGPU.jar
	zipmerge $@ $(SCALA_ROOT)/scala-library.jar gpu/MontePiGPU.jar $<
