#!/bin/bash
rm *.class *.jar; javac *.java -classpath ../rootbeer1/Rootbeer.jar && jar -cvfm TestMergeSort.jar manifest.txt *.class && java -jar ../rootbeer1/Rootbeer.jar TestMergeSort.jar TestMergeSortGPU.jar -64bit -computecapability=sm_30
# scp -r -P 2222 . hypatia@141.30.223.168:"/media/d/Studium/9TH\ SEMESTER/Hauptseminar\ CUDA\ on\ Sparc/MergeSortGPU/"
