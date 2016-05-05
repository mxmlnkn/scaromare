.phony: clean all

all: TestMonteCarloPiV2.exe run
run:
	./TestMonteCarloPiV2.exe 268435456
clean:
	rm -f *.exe *.log *.o

TestMonteCarloPiV2.exe: TestMonteCarloPiV2.cpp
	nvcc -arch=sm_30 -x cu $< -o $@ -std=c++11 -DNDEBUG -O3
