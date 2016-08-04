
.phony: clean all *.class *.jar


all: MontePi.jar

run:
	java -jar MontePi.jar 268435456

clean:
	rm -f *.class *.jar


MontePi.jar: manifest.txt TestMonteCarloPi.class MonteCarloPi.class MonteCarloPiKernel.class
	jar cfm $@ $^

%.class: %.java
	javac -classpath . $<
