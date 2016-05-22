ROOTBEER_ROOT := ../../../../rootbeer1

.phony: clean all MontePiGPU.jar *.class

all: MontePi.jar run
run:
	java -jar MontePi.jar 268435456
clean:
	rm -f *.class *.jar *.tmp

MonteCarloPiKernel.class: MonteCarloPiKernel.java
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
MonteCarloPi.class: MonteCarloPi.java MonteCarloPiKernel.class
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
TestMonteCarloPi.class: TestMonteCarloPi.java MonteCarloPi.class
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.

# Splitting compilation in these units makes it faster, because Rootbeer is very slow!
gpu.jar: MonteCarloPiKernel.class
	# Note that .tmp is not possible, because Soot won't analyze files not ending with .jar!
	jar -cvf $@.tmp.jar manifest.txt $^
	java -jar $(ROOTBEER_ROOT)/Rootbeer.jar $@.tmp.jar $@ -64bit -computecapability=sm_30
cpu.jar: TestMonteCarloPi.class MonteCarloPi.java
	jar -cvfm $@ manifest.txt $^

# Order of dependencies is important, because latter overwrites former especially the manifest file!
MontePi.jar: gpu.jar cpu.jar
	zipmerge MontePi.jar $(ROOTBEER_ROOT)/Rootbeer.jar $^
