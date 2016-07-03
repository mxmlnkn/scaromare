ROOTBEER_ROOT := ../../../../../rootbeer1
ROOTBEER_JAR   = $(ROOTBEER_ROOT)/Rootbeer.jar

.phony: clean all

all: gpu.jar

clean:
	rm -f *.class *.jar

MonteCarloPiKernel.class: MonteCarloPiKernel.java
	javac $< -classpath $(ROOTBEER_JAR):.

gpu.jar: MonteCarloPiKernel.class
	jar -cvf $@.tmp.jar $^
	java -jar $(ROOTBEER_JAR) $@.tmp.jar $@ -64bit -computecapability=sm_30
