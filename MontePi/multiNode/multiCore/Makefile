
SPARK_VERSION ?= 1.5.2
SPARK_ROOT    ?= /opt/spark-$(SPARK_VERSION)
SCALA_ROOT    ?= /usr/share/java
SCALA_JAR     ?= $(SCALA_ROOT)/scala-library.jar
SPARK_JAR     ?= $(SPARK_ROOT)/core/target/spark-core_2.10-$(SPARK_VERSION).jar


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
	scalac -classpath $(SPARK_JAR):. -deprecation $<


MontePi.jar: manifest.txt TestMonteCarloPi.class
	jar -cvfm $@.tmp.jar $< *.class
	zipmerge $@ $(SCALA_JAR) $@.tmp.jar
