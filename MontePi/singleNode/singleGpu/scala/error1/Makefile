ROOTBEER_ROOT := ../../../../../rootbeer1

.phony: clean all MontePiGPU.jar *.class

all: MontePiGPU.jar run
run: MontePiGPU.jar
	scala MontePiGPU.jar
clean:
	rm -f *.class *.jar

MonteCarloPiKernel.class: MonteCarloPiKernel.scala
	scalac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:. -deprecation
MonteCarloPi.class: MonteCarloPi.java MonteCarloPiKernel.class
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
TestMonteCarloPi.class: TestMonteCarloPi.scala MonteCarloPi.class
	scalac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:. -deprecation

MontePiGPU.jar: TestMonteCarloPi.class manifest.txt
	jar -cvfm $@.tmp.jar manifest.txt *.class
	java -jar $(ROOTBEER_ROOT)/Rootbeer.jar $@.tmp.jar $@ -64bit -computecapability=sm_30
