ROOTBEER_ROOT := ../../../../../rootbeer1
SCALA_JAR     := /usr/share/java/scala-library.jar

.phony: clean all MontePiGPU.jar *.class

all: MontePiGPU.jar run
run: MontePiGPU.jar
	#scala MontePiGPU.jar
	java -jar MontePiGPU.jar.merged
clean:
	rm -f *.class *.jar *.tmp *.merged

DummyEntry.class: DummyEntry.java
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
MonteCarloPiKernel.class: MonteCarloPiKernel.java
	#scalac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:. -deprecation
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
MonteCarloPi.class: MonteCarloPi.java MonteCarloPiKernel.class
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
TestMonteCarloPi.class: TestMonteCarloPi.scala MonteCarloPi.class
	scalac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:. -deprecation

MontePiGPU.jar: TestMonteCarloPi.class DummyEntry.class manifest.txt
	# put everything into $@.tmp plus a manifest pointing to the dummy entry
	# and compile it with Rootbeer
	jar -cfm tmp.jar manifest.txt *.class
	# can't just merge into $@.tmp, as then META-INF/MANIFEST.MF from
	# SCALA_JAR would overwrite that from $@.tmp!
	zipmerge merged.jar $(ROOTBEER_ROOT)/Rootbeer.jar $(SCALA_JAR) tmp.jar
	java -jar $(ROOTBEER_ROOT)/Rootbeer.jar merged.jar $@ -64bit -computecapability=sm_30
	# Now replace the manifest point to the Dummy, and also delete the dummy
	#zipmerge $@ $(SCALA_JAR) gpu/MontePiGPU.jar $<
	zip --update META-INF/MANIFEST.MF
	zip
