
SCALA_ROOT    ?= /usr/share/java
SCALA_JAR     ?= $(SCALA_ROOT)/scala-library.jar


.phony: clean all *.class *.jar

all: MontePi.jar

run:
	java -jar MontePi.jar 268435456

clean:
	rm -f *.class *.jar


MontePi.jar: manifest.txt TestMonteCarloPi.class
	jar cfm $@.tmp $< *.class
	zipmerge $@ $(SCALA_JAR) $@.tmp

TestMonteCarloPi.class: TestMonteCarloPi.scala
	scalac -classpath . $< -deprecation
