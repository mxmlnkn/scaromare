#!/bin/bash

make -C MontePi/singleNode/singleCore/java/
make -C MontePi/singleNode/singleCore/scala/
make -C MontePi/singleNode/singleGpu/cpp/
make -C MontePi/singleNode/singleGpu/java/
make -C MontePi/singleNode/multiGpu/java/
make -C MontePi/singleNode/multiGpu/scala/
make -C MontePi/multiNode/multiCore/
#make -C MontePi/multiNode/multiGpu/java/
make -C MontePi/multiNode/multiGpu/scala/


git checkout MontePi/multiNode/multiGpu/scala/gpu/MonteCarloPiKernel.java
git checkout MontePi/singleNode/multiGpu/java/MonteCarloPiKernel.java
git checkout MontePi/singleNode/multiGpu/scala/MonteCarloPi.java
git checkout MontePi/singleNode/multiGpu/scala/MonteCarloPiKernel.java
git checkout MontePi/singleNode/singleGpu/java/MonteCarloPiKernel.java
