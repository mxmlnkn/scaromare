ROOTBEER_ROOT := ../../../../rootbeer1
SCALA_ROOT    := /usr/share/java

.phony: clean all MontePiGPU.jar *.class

all: MontePi.jar run
run: MontePi.jar
	#java -jar MontePi.jar 2684354560
clean:
	rm -f *.class *.jar

MonteCarloPiKernel.class: MonteCarloPiKernel.java
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
MonteCarloPi.class: MonteCarloPi.java MonteCarloPiKernel.class
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
TestMonteCarloPi.class: TestMonteCarloPi.scala MontePiGPU.jar
	scalac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:. -deprecation

gpu.jar: MonteCarloPiKernel.class
	jar -cf tmp.jar $^
	java -jar $(ROOTBEER_ROOT)/Rootbeer.jar tmp.jar $@ -64bit -computecapability=sm_30

nongpu.jar: MonteCarloPi.class TestMonteCarloPi.class
	jar -cfm $@ manifest.txt $^ TestMonteCarloPi$*.class MonteCarloPi$*.class
	# Watch out! MonteCarloPi*.class would also encompass MonteCarloPiKernel.class which was not anaylzed by Rootbeer, thereby possibly giving problems because of clashing!
	#	Exception in thread "main" java.lang.ClassCastException: MonteCarloPiKernel cannot be cast to org.trifort.rootbeer.runtime.CompiledKernel
	#		at org.trifort.rootbeer.runtime.CUDAContext.setKernel(CUDAContext.java:119)
	#		at MonteCarloPi.runOnDevice(MonteCarloPi.java:116)
	#		at MonteCarloPi.calc(MonteCarloPi.java:180)
	#		at TestMonteCarloPi$.main(TestMonteCarloPi.scala:75)
	#		at TestMonteCarloPi.main(TestMonteCarloPi.scala)
	# But the * is needed if it should be tried to use MonteCarlo.scala because of those automatically created classes like MonteCarloPi$1.class
	# Either do: MonteCarloPi.class MonteCarloPi$*.class or
	# take care that gpu.jar is behind nongpu.jar which means it will override nongpu.jar

MontePi.jar: nongpu.jar gpu.jar
	zipmerge $@ $(SCALA_ROOT)/scala-library.jar $^

# jar -cf tmp.jar MonteCarloPiKernel.class &&
# java -jar "$ROOTBEER_ROOT"/Rootbeer.jar tmp.jar gpu.jar -64bit -computecapability=sm_30 &&
# jar -cfm nongpu.jar manifest.txt TestMonteCarloPi*.class MonteCarloPi.class &&
# zipmerge MontePi.jar "$SCALA_JAR" gpu.jar nongpu.jar &&
# java -jar MontePi.jar 2684354560
