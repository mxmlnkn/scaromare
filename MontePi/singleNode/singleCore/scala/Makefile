SCALA_ROOT    := /usr/share/java

.phony: clean all *.class *.jar

all: MontePi.jar run

run:
	java -jar MontePi.jar 268435456

clean:
	rm -f *.class *.jar

MontePi.jar: TestMonteCarloPi.class
	jar cfm $@.tmp manifest.txt *.class
	zipmerge $@ $(SCALA_ROOT)/scala-library.jar $@.tmp

TestMonteCarloPi.class: TestMonteCarloPi.scala
	scalac -classpath . $< -deprecation
