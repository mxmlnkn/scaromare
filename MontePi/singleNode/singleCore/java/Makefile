
.phony: clean all *.class *.jar

all: MontePi.jar run

run:
	java -jar MontePi.jar 268435456
clean:
	rm -f *.class *.jar

MontePi.jar: TestMonteCarloPi.class MonteCarloPi.class MonteCarloPiKernel.class
	jar cfm $@ manifest.txt $*

%.class: %.java
	javac -classpath . $<
