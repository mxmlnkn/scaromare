ROOTBEER_ROOT := ../../../../rootbeer1
SCALA_ROOT    := /usr/share/java

.phony: clean all MontePiGPU.jar *.class

all: MontePi.jar run
run: MontePi.jar
	scala MontePiGPU.jar 2684354560
clean:
	rm -f *.class *.jar

MonteCarloPiKernel.class: MonteCarloPiKernel.java
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
MonteCarloPi.class: MonteCarloPi.java MonteCarloPiKernel.class
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
DummyEntry.class: DummyEntry.java
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
MontePiGPU.jar: MonteCarloPiKernel.class MonteCarloPi.class DummyEntry.class
	jar -cvfm MontePiGPUtmp.jar manifest.txt MonteCarloPiKernel.class MonteCarloPi.class DummyEntry.class
	java -jar $(ROOTBEER_ROOT)/Rootbeer.jar MontePiGPUtmp.jar MontePiGPU.jar -64bit -computecapability=sm_30

TestMonteCarloPi.class: TestMonteCarloPi.scala MontePiGPU.jar
	scalac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:. -deprecation
TestMontePi.jar: TestMonteCarloPi.class
	jar -cvfm $@ manifest.txt *.class

MontePi.jar: TestMontePi.jar
	zipmerge $@ $(SCALA_ROOT)/scala-library.jar MontePiGPU.jar $<

