SCALA_FILES   := $(wildcard *.scala)
CLASS_FILES   := $(SCALA_FILES:.scala=.class)
ROOTBEER_ROOT := ../../rootbeer1
SPARK_ROOT    := /opt/spark-1.5.2
SCALA_ROOT    := /usr/share/java/

.phony: clean all *.class *.jar

all: JavaSparkPiTmp.jar
	# merge dependencies (Rootbeer,Scala) into a fat jar
	# Note that the rightmost jar overwrites all to the left, this is
	# important for the manifest, because we want to use the manifest
	# of JavaSparkPiTmp.jar
	zipmerge JavaSparkPi.jar $(SPARK_ROOT)/core/target/spark-core_2.10-1.5.2.jar $(SCALA_ROOT)scala-library.jar JavaSparkPiTmp.jar
	$(SPARK_ROOT)/bin/spark-submit --class JavaSparkPi JavaSparkPi.jar

clean:
	rm -f *.class *.jar

JavaSparkPiTmp.jar: JavaSparkPi.class
	jar cfm $@ manifest.txt *.class
JavaSparkPi.class: JavaSparkPi.java
	javac -cp $(SPARK_ROOT)/core/target/spark-core_2.10-1.5.2.jar:. $<
