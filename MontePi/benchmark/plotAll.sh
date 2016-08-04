#!/bin/bash

python plot.py "./raw-logs/benchmarks-2016-05-02_01-40-04/results.log" . "./raw-logs/benchmarks-2016-05-02_01-40-04/error-scaling.log"

# ../../plot.py -e ./raw-logs/cpp-old-pin.log ./raw-logs/python-pin.dat ./raw-logs/cpp-pin.log ./raw-logs/cpp-pin-237890291.log ./raw-logs/cpp-pin-684168515.log
python plot.py -e ./raw-logs/cpp-pin-237890291.log 'Messung für Seed 237890291' ./raw-logs/cpp-pin-684168515.log 'Messung für Seed 684168515'

python plot.py -s strong-scaling_2016-08-04_03-40-03 -k weak-scaling_2016-08-03_22-52-00
