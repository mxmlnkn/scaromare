
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


MonteCarloPiKernel.class: MonteCarloPiKernel.scala
	scalac $< -classpath $(ROOTBEER_JAR):. -deprecation

MonteCarloPi.class: MonteCarloPi.java MonteCarloPiKernel.class
# The error
# ./MonteCarloPiKernel.class: warning: Cannot find annotation method 'bytes()' in type 'ScalaSignature': class file for scala.reflect.ScalaSignature not found
# happens because this classes uses MonteCarloPiKernel which in turn
# uses function from SCALA_JAR, so it needs to be in the classpath!
	javac $< -classpath $(ROOTBEER_JAR):$(SCALA_JAR):.
#javac $< -classpath $(ROOTBEER_JAR):.

TestMonteCarloPi.class: TestMonteCarloPi.scala MonteCarloPi.class
	scalac $< -classpath . -deprecation


MontePi.jar: manifest.txt TestMonteCarloPi.class
	jar -cvfm gpu.jar $< *.class
# The error
# java.lang.RuntimeException: cannot get resident body for phantom class : <MonteCarloPiKernel$$anonfun$gpuMethod$1: void <init>(MonteCarloPiKernel,scala.runtime.ObjectRef)>; maybe you want to call c.setApplicationClass() on this class!
# happens because Soot couldn't find a function call. That is because
# SCALA_JAR was not merged into $@.tmp.jar, doing that solves this error!
	zipmerge $@.tmp.jar $(SCALA_JAR) gpu.jar
# old version with error directly compiled gpu.jar
	java -jar $(ROOTBEER_COMPILER) -64bit -nofatjar $@.tmp.jar $@


# @todo The error remaining is a memory overflow on "loading reverse string call graph",
# maybe the scala kernel somehow results in a cyclic graph which in turn
# results in the memory leak ???
#
#   loading reverse string call graph for: <MonteCarloPiKernel: void gpuMethod()>...
#   slurmstepd: Step 7771122.1 exceeded memory limit (5671620 > 4096000), being killed
#   salloc: Exceeded job memory limit
#   slurmstepd: *** STEP 7771122.1 ON taurusi2107 CANCELLED AT 2016-08-04T16:36:15 ***
#   srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
#   srun: error: taurusi2107: task 0: Killed
#
# Increasing to 12GB memory solves this, but results in other errors:
#
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
