.phony: clean all

all: TestMonteCarloPi.exe run
run:
	./TestMonteCarloPi.exe 268435456
clean:
	rm -f *.exe *.log *.o

TestMonteCarloPi.exe: TestMonteCarloPi.cpp
	nvcc -x cu $< -o $@ -std=c++11
