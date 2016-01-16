SCALA_FILES   := $(wildcard *.scala)
CLASS_FILES   := $(SCALA_FILES:.scala=.class)
ROOTBEER_ROOT := ../../rootbeer1/

.phony: clean all *.class

all: *.class
	$(MAKE) -C gpu

	# copy and unpack dependencies (Rootbeer,Scala)
	mkdir -p swtge878t1
	cd swtge878t1                               && \
		cp /usr/share/java/scala-library.jar .  && \
		cp ../gpu/MontePiGPU.jar .              && \
		cp ../manifest.txt ../*.class .         && \
		jar xf scala-library.jar                && \
		jar xf MontePiGPU.jar                   && \
		jar cfm MontePi.jar manifest.txt *      && \
		./MontePi.jar

clean:
	rm -f *.class *.jar

%.class: %.scala
	scalac -classpath gpu/MontePiGPU.jar:. $< -deprecation
