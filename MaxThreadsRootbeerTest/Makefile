ROOTBEER_ROOT := ../../../../rootbeer1

.phony: clean all MontePiGPU.jar *.class

all: MontePi.jar run
run:
	java -jar MontePi.jar 268435456
clean:
	rm -f *.class *.jar

MonteCarloPi.class: MonteCarloPi.java
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
MonteCarloPiKernel.class: MonteCarloPiKernel.java MonteCarloPi.class
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
TestMonteCarloPi.class: TestMonteCarloPi.java MonteCarloPiKernel.class
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.

MontePi.jar: TestMonteCarloPi.class
	jar -cvfm $@.tmp.jar manifest.txt *.class
	java -jar $(ROOTBEER_ROOT)/Rootbeer.jar $@.tmp.jar $@ -64bit -computecapability=sm_30

