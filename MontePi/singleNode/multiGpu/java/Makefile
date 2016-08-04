
ROOTBEER_ROOT      := ../../../../rootbeer1
ROOTBEER_COMPILER   = $(ROOTBEER_ROOT)/Rootbeer.jar
ROOTBEER_JAR        = $(ROOTBEER_ROOT)/RootbeerRuntime.jar
SCALA_JAR           = $(SCALA_ROOT)/scala-library.jar


.phony: clean all MontePiGPU.jar *.class

all: MontePi.jar run
run:
	java -jar MontePi.jar 268435456
clean:
	rm -f *.class *.jar *.tmp


MonteCarloPi.class: MonteCarloPi.java $(ROOTBEER_JAR)
	javac $< -classpath $(ROOTBEER_JAR):.
MonteCarloPiKernel.class: MonteCarloPiKernel.java MonteCarloPi.class $(ROOTBEER_JAR)
	javac $< -classpath $(ROOTBEER_JAR):.
TestMonteCarloPi.class: TestMonteCarloPi.java MonteCarloPiKernel.class
	javac $< -classpath .


gpu.jar: MonteCarloPiKernel.class
	jar -cf gputmp.jar $^
	java -jar $(ROOTBEER_COMPILER) -64bit -nofatjar gputmp.jar $@

cpu.jar: manifest.txt MonteCarloPi.class TestMonteCarloPi.class
	jar -cfm $@ $^

MontePi.jar: $(SCALA_JAR) $(ROOTBEER_JAR) gpu.jar cpu.jar
	zipmerge $@ $^

