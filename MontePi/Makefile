SCALA_FILES   := $(wildcard *.scala)
CLASS_FILES   := $(SCALA_FILES:.scala=.class)
ROOTBEER_ROOT := ../../rootbeer1/

.phony: clean all *.class

all: *.class
	$(MAKE) -C gpu

	# merge dependencies (Rootbeer,Scala) into a fat jar
	# Note that the rightmost jar overwrites all to the left, this is
	# important for the manifest, because we want to use the manifest
	# of TestMonteCarloPi.jar
	zipmerge MontePi.jar gpu/MontePiGPU.jar /usr/share/java/scala-library.jar TestMonteCarloPi.jar
	java -jar MontePi.jar

clean:
	$(MAKE) -C gpu clean
	rm -f *.class *.jar

TestMonteCarloPi.jar: TestMonteCarloPi.class
	jar cfm TestMonteCarloPi.jar manifest.txt *.class

TestMonteCarloPi.class: TestMonteCarloPi.scala
	scalac -classpath gpu/MontePiGPU.jar:. $< -deprecation
