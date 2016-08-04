
ROOTBEER_ROOT      ?= ../../../../rootbeer1
SCALA_ROOT         ?= /usr/share/java
ROOTBEER_COMPILER  ?= $(ROOTBEER_ROOT)/Rootbeer.jar
ROOTBEER_JAR       ?= $(ROOTBEER_ROOT)/RootbeerRuntime.jar
SCALA_JAR          ?= $(SCALA_ROOT)/scala-library.jar


.phony: clean all run MontePi.jar *.class

all: MontePi.jar

run: MontePi.jar
	java -jar MontePi.jar 268435456 1

clean:
	rm -f *.class *.jar


MonteCarloPiKernel.class: MonteCarloPiKernel.java
	javac $< -classpath $(ROOTBEER_JAR):.

Distribute.class: Distribute.scala
	scalac $< -classpath . -deprecation

MonteCarloPi.class: MonteCarloPi.scala MonteCarloPiKernel.class Distribute.class
	scalac $< -classpath $(ROOTBEER_JAR):. -deprecation

TestMonteCarloPi.class: TestMonteCarloPi.scala
	scalac $< -classpath $(ROOTBEER_JAR):. -deprecation


gpu.jar: MonteCarloPiKernel.class
	jar -cf gputmp.jar $^
	java -jar $(ROOTBEER_COMPILER) -64bit -nofatjar gputmp.jar $@

cpu.jar: MonteCarloPi.class TestMonteCarloPi.class
	jar -cfm $@ manifest.txt $^ TestMonteCarloPi$*.class MonteCarloPi$*.class


MontePi.jar: $(SCALA_JAR) $(ROOTBEER_JAR) gpu.jar cpu.jar
	zipmerge $@ $^
