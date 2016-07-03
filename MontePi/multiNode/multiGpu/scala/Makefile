# on Taurus use:
# make SPARK_ROOT=~/spark-1.5.2-bin-hadoop2.6 SPARK_JAR=~/spark-1.5.2-bin-hadoop2.6/lib/spark-assembly-1.5.2-hadoop2.6.0.jar SCALA_ROOT=$(dirname $(which scala))/../lib
ROOTBEER_ROOT := ../../../../rootbeer1
SPARK_ROOT    := /opt/spark-1.5.2
SCALA_ROOT    := /usr/share/java
SCALAC        := scalac
JAVAC         := javac
ROOTBEER_JAR   = $(ROOTBEER_ROOT)/Rootbeer.jar
SCALA_JAR	   = $(SCALA_ROOT)/scala-library.jar
SPARK_JAR      = $(SPARK_ROOT)/assembly/target/scala-2.10/spark-assembly-1.5.2-hadoop2.2.0.jar

.phony: clean all *.jar

all: MontePi.jar
run: MontePi.jar
	$(SPARK_ROOT)/bin/spark-submit --master local[4] --class TestMonteCarloPi $< 268435456 1
clean:
	$(MAKE) -C gpu clean
	rm -f *.class *.jar

MonteCarloPi.class: MonteCarloPi.java gpu/gpu.jar
	$(MAKE) -C gpu gpu.jar
	javac $< -classpath $(ROOTBEER_JAR):gpu/gpu.jar

TestMonteCarloPi.class: TestMonteCarloPi.scala MonteCarloPi.class
	$(SCALAC) $< -classpath $(ROOTBEER_JAR):$(SPARK_JAR):cpu.jar:gpu/gpu.jar:. -deprecation

cpu.jar: TestMonteCarloPi.class MonteCarloPi.class
	jar -cvfm $@ manifest.txt $^

MontePi.jar: gpu/gpu.jar cpu.jar
	zipmerge $@ $(ROOTBEER_JAR) $(SCALA_JAR) $^
