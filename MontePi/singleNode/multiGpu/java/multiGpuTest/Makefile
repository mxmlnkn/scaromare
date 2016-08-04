
ROOTBEER_ROOT     ?= ../../../../rootbeer1
ROOTBEER_COMPILER ?= $(ROOTBEER_ROOT)/Rootbeer.jar
ROOTBEER_JAR      ?= $(ROOTBEER_ROOT)/RootbeerRuntime.jar


.phony: clean all MontePiGPU.jar *.class

all: MontePi.jar

run:
	java -jar MontePi.jar 2684354

clean:
	rm -f *.class *.jar *.tmp


MonteCarloPiKernel.class: MonteCarloPiKernel.java
	javac $< -classpath $(ROOTBEER_JAR):.

MonteCarloPi.class: MonteCarloPi.java MonteCarloPiKernel.class
	javac $< -classpath $(ROOTBEER_JAR):.

TestMonteCarloPi.class: TestMonteCarloPi.java MonteCarloPi.class
	javac $< -classpath .


gpu.jar: MonteCarloPiKernel.class
	jar -cvf $@.tmp.jar $^
	java -jar $(ROOTBEER_COMPILER) -64bit -nofatjar $@.tmp.jar $@

cpu.jar: manifest.txt TestMonteCarloPi.class MonteCarloPi.java
	jar -cvfm $@ $^


MontePi.jar: $(ROOTBEER_JAR) gpu.jar cpu.jar
	zipmerge MontePi.jar $^
