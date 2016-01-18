SCALA_FILES   := $(wildcard *.scala)
CLASS_FILES   := $(SCALA_FILES:.scala=.class)
ROOTBEER_ROOT := ../../rootbeer1
SPARK_ROOT    := /opt/spark-1.5.2
TMP_FOLDER    := swtge878t1

.phony: clean all *.class

all: JavaSparkPi.class
	# copy and unpack dependencies (Rootbeer,Scala)
	mkdir -p $(TMP_FOLDER)
		cp $(SPARK_ROOT)/core/target/spark-core_2.10-1.5.2.jar $(TMP_FOLDER)/spark-core.jar
		cp /usr/share/java/scala-library.jar $(TMP_FOLDER)/
		cp *.class manifest.txt $(TMP_FOLDER)/

	cd swtge878t1                && \
		jar xf spark-core.jar    && \
		jar xf scala-library.jar && \
		rm *.jar                 && \
		jar cfm MontePiSpark.jar manifest.txt *

clean:
	rm -f *.class *.jar

JavaSparkPi.class: JavaSparkPi.java
	javac -cp $(SPARK_ROOT)/core/target/spark-core_2.10-1.5.2.jar:. $<
