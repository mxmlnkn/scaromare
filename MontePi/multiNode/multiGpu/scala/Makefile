# On Taurus:
#   makeOpts=( \
#       "SPARK_ROOT=$HOME/spark-1.5.2-bin-hadoop2.6" \
#       "SPARK_JAR=$HOME/spark-1.5.2-bin-hadoop2.6/lib/spark-assembly-1.5.2-hadoop2.6.0.jar" \
#       "SPARKCORE_JAR=$SPARK_JAR" \
#       "SCALA_ROOT=$(dirname $(which scala))/../lib" \
#   )
# make ${makeOpts[@]} MontePi.jar

ROOTBEER_ROOT := ../../../../rootbeer1
SPARK_ROOT    := /opt/spark-1.5.2
SCALA_ROOT    := /usr/share/java
SCALAC        := scalac
JAVAC         := javac
ROOTBEER_JAR   = $(ROOTBEER_ROOT)/Rootbeer.jar
SPARK_JAR      = $(SPARK_ROOT)/core/target/spark-core_2.10-1.5.2.jar
SCALA_JAR      = $(SCALA_ROOT)/scala-library.jar

.phony: clean all *.jar *.class

all: MontePi.jar
run: MontePi.jar
	$(SPARK_ROOT)/bin/spark-submit --master local[4] --class TestMonteCarloPi $< 268435456 1
clean:
	$(MAKE) -C gpu clean
	rm -f *.class *.jar

TestMonteCarloPi.class: TestMonteCarloPi.scala
	$(MAKE) -C gpu MontePiGPU.jar
	$(SCALAC) $< -classpath $(SPARK_JAR):$(ROOTBEER_JAR):gpu/MontePiGPU.jar:. -deprecation

TestMontePi.jar: TestMonteCarloPi.class
	jar -cvfm $@ manifest.txt *.class

MontePi.jar: $(SCALA_JAR) $(ROOTBEER_JAR) gpu/MontePiGPU.jar TestMontePi.jar
	$(MAKE) -C gpu MontePiGPU.jar
	zipmerge $@ $^
