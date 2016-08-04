
ROOTBEER_ROOT      ?= ../../../../rootbeer1
SCALA_ROOT         ?= /usr/share/java
ROOTBEER_COMPILER  ?= $(ROOTBEER_ROOT)/Rootbeer.jar
ROOTBEER_JAR       ?= $(ROOTBEER_ROOT)/RootbeerRuntime.jar
SCALA_JAR          ?= $(SCALA_ROOT)/scala-library.jar


.phony: clean all MontePi.jar *.class


all: MontePi.jar

run: MontePi.jar
	scala MontePi.jar

clean:
	rm -f *.class *.jar


MonteCarloPiKernel.class: MonteCarloPiKernel.java
	javac $< -classpath $(ROOTBEER_JAR):.

MonteCarloPi.class: MonteCarloPi.scala MonteCarloPiKernel.class
	scalac $< -classpath $(ROOTBEER_JAR):. -deprecation

TestMonteCarloPi.class: TestMonteCarloPi.scala MonteCarloPi.class
	scalac $< -classpath . -deprecation


MontePi.jar: manifest.txt TestMonteCarloPi.class $(ROOTBEER_JAR)
	jar -cvfm gpu.jar $< *.class
# this step was left out, therefore phantom class error happened, but
# including this will result in a memory overrun, because Soot tries
# to analyze the whole SCALA_JAR library:
# loading reverse string call graph for: <MonteCarloPiKernel: void gpuMethod()>...
# slurmstepd: Step 7771122.3 exceeded memory limit (5630396 > 4096000), being killed
# Increasing to 12GB memory solves this ...
	zipmerge $@.tmp.jar $(SCALA_JAR) gpu.jar
	java -jar $(ROOTBEER_COMPILER) -64bit -nofatjar $@.tmp.jar $@.compiled.jar
	zipmerge $@ $(ROOTBEER_JAR) $(SCALA_JAR) $@.compiled.jar
# Running scala $@ results in
# java.lang.ClassNotFoundException: org.trifort.rootbeer.runtime.Kernel#
# not sure why, because zipinfo $@ shows org/trifort/rootbeer/runtime/Kernel.class
#
# Running with java -jar $@ results in
# Exception in thread "main" java.lang.NoSuchMethodError: MonteCarloPi: method <init>()V not found
# 	at TestMonteCarloPi$.main(TestMonteCarloPi.scala:8)
# 	at TestMonteCarloPi.main(TestMonteCarloPi.scala)
# Also not sure why, I have newer other tests where running with java -jar works, because $(SCALA_JAR) is compiled into the resulting file, but that should be the case here too (although it is run through Rootbeer in contrast to the working tests)
#
# The newer tests avoid running Rootbeer over SCALA_JAR, because it is not necessary as long as the kernel is writting in java, as long as only the kernel is merged into a jar and then given to Rootbeer.
#
# It could also be worth experimenting with the -libjar option to Rootbeer, e.g. -libjar $(SCALA_JAR) ??
