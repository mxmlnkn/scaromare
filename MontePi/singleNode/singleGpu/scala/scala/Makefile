# For Taurus:
#    make -B SPARK_ROOT=~/spark-1.5.2-bin-hadoop2.6 SPARKCORE_JAR=~/spark-1.5.2-bin-hadoop2.6/lib/spark-assembly-1.5.2-hadoop2.6.0.jar SCALA_ROOT=$(dirname $(which scala))/../lib

ROOTBEER_ROOT := ../../../../../rootbeer1
SPARK_ROOT    := /opt/spark-1.5.2
SCALA_ROOT    := /usr/share/java
SCALAC        := scalac
JAVAC         := javac

SPARKCORE_JAR = $(SPARK_ROOT)/core/target/spark-core_2.10-1.5.2.jar
HADOOP_JAR    = $(SPARK_ROOT)/assembly/target/scala-2.10/spark-assembly-1.5.2-hadoop2.2.0.jar

.phony: clean all *.jar

all: MontePi.jar
run: MontePi.jar
	$(SPARK_ROOT)/bin/spark-submit --master local[1] --class TestMonteCarloPi $< 268435456 1
clean:
	rm -f *.class *.jar


MonteCarloPi.class: MonteCarloPi.scala
	#javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
	$(SCALAC) $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:$(HADOOP_JAR):$(SPARKCORE_JAR):MontePiGPU.jar:. -deprecation
MonteCarloPiKernel.class: MonteCarloPiKernel.java
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.

MontePiGPU.jar: MonteCarloPiKernel.class
	jar -cvf MontePiGPUtmp.jar $^
	java -jar $(ROOTBEER_ROOT)/Rootbeer.jar MontePiGPUtmp.jar $@ -64bit -computecapability=sm_30


TestMonteCarloPi.class: TestMonteCarloPi.scala MonteCarloPi.class MontePiGPU.jar
	$(SCALAC) $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:$(HADOOP_JAR):$(SPARKCORE_JAR):MontePiGPU.jar:. -deprecation
MontePiCPU.jar: TestMonteCarloPi.class MonteCarloPi.class
	# Note that \$$ -> \$ by Makefile and \$ -> $ by interpreting shell
	jar -cvfm $@ manifest.txt $^ TestMonteCarloPi\$$*.class MonteCarloPi\$$*.class
MontePi.jar: MontePiGPU.jar MontePiCPU.jar
	zipmerge $@ $(ROOTBEER_ROOT)/Rootbeer.jar $(SCALA_ROOT)/scala-library.jar $^
