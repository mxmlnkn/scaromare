JAVA_FILES    := $(wildcard *.java)
CLASS_FILES   := $(JAVA_FILES:.java=.class)
ROOTBEER_ROOT := ../../../../../rootbeer1

.phony: clean all MontePiGPU.jar $(CLASS_FILES)

all: $(CLASS_FILES) MontePiGPU.jar

clean:
	rm -f *.class *.jar

MonteCarloPi.class: MonteCarloPi.java
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
MonteCarloPiKernel.class: MonteCarloPiKernel.java
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
DummyEntry.class: DummyEntry.java
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.

MontePiGPU.jar: $(CLASS_FILES)
	jar -cvfm MontePiGPUtmp.jar manifest.txt *.class
	java -jar $(ROOTBEER_ROOT)/Rootbeer.jar MontePiGPUtmp.jar MontePiGPU.jar -64bit -computecapability=sm_30

