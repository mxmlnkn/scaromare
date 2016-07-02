ROOTBEER_ROOT := ../../../../rootbeer1

.phony: clean all MontePiGPU.jar *.class

all: MontePi.jar run
run:
	java -jar MontePi.jar 268435456
clean:
	rm -f *.class *.jar *.tmp

MonteCarloPi.class: MonteCarloPi.java $(ROOTBEER_ROOT)/Rootbeer.jar
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
MonteCarloPiKernel.class: MonteCarloPiKernel.java MonteCarloPi.class $(ROOTBEER_ROOT)/Rootbeer.jar
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
TestMonteCarloPi.class: TestMonteCarloPi.java MonteCarloPiKernel.class $(ROOTBEER_ROOT)/Rootbeer.jar
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.

gpu.jar: MonteCarloPiKernel.class
	jar -cf tmp.jar $^
	java -jar $(ROOTBEER_ROOT)/Rootbeer.jar tmp.jar $@ -64bit -computecapability=sm_30
nongpu.jar: MonteCarloPi.class TestMonteCarloPi.class
	jar -cfm $@ manifest.txt $^

MontePi.jar: gpu.jar nongpu.jar
	zipmerge $@ $(ROOTBEER_ROOT)/Rootbeer.jar $(SCALA_ROOT)/scala-library.jar $^

