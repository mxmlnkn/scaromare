
ROOTBEER_ROOT      ?= ../../../../rootbeer1
SCALA_ROOT         ?= /usr/share/java
ROOTBEER_COMPILER  ?= $(ROOTBEER_ROOT)/Rootbeer.jar
ROOTBEER_JAR       ?= $(ROOTBEER_ROOT)/RootbeerRuntime.jar
SCALA_JAR          ?= $(SCALA_ROOT)/scala-library.jar


.phony: clean all MontePi.jar *.class


all: MontePi.jar

run: MontePi.jar
#scala MontePiGPU.jar
	java -jar MontePi.jar.merged

clean:
	rm -f *.class *.jar *.tmp *.merged


DummyEntry.class: DummyEntry.java
	javac $< -classpath .

MonteCarloPiKernel.class: MonteCarloPiKernel.java
#scalac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:. -deprecation
	javac $< -classpath $(ROOTBEER_JAR):.

MonteCarloPi.class: MonteCarloPi.java MonteCarloPiKernel.class
	javac $< -classpath $(ROOTBEER_JAR):.

TestMonteCarloPi.class: TestMonteCarloPi.scala MonteCarloPi.class
	scalac $< -classpath $(ROOTBEER_JAR):. -deprecation


MontePi.jar: TestMonteCarloPi.class DummyEntry.class manifest.txt
# put everything into $@.tmp plus a manifest pointing to the dummy entry
# and compile it with Rootbeer
# isn't this problematic because of MonteCarloPiKernel.class ???
	jar -cfm tmp.jar manifest.txt *.class
# can't just merge into $@.tmp, as then META-INF/MANIFEST.MF from
# SCALA_JAR would overwrite that from $@.tmp!
	zipmerge merged.jar $(ROOTBEER_JAR) $(SCALA_JAR) tmp.jar
	java -jar $(ROOTBEER_COMPILER) -64bit -nofatjar merged.jar $@
# Now replace the manifest point to the Dummy, and also delete the dummy
#zipmerge $@ $(SCALA_JAR) gpu/MontePi.jar $<
	zip --update META-INF/MANIFEST.MF
	zip
