javac ThreadIDsKernel.java -classpath "$ROOTBEER_ROOT/Rootbeer.jar:." &&
javac ThreadIDs.java       -classpath "$ROOTBEER_ROOT/Rootbeer.jar:." &&
cat > manifest.txt <<EOF
Main-Class: ThreadIDs
Class-Path: .
EOF
jar -cvfm preRootbeer.tmp.jar manifest.txt ThreadIDsKernel.class ThreadIDs.class &&
java -jar "$ROOTBEER_ROOT/Rootbeer.jar" preRootbeer.tmp.jar ThreadIDs.jar -64bit &&
java -jar ThreadIDs.jar
