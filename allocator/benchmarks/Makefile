CC=nvcc
UTIL=../../utilities
FOLDERS=-I ../source_files -I/usr/local -L/usr/lib64 -I $(UTIL)
LINKS= -lpthread -lboost_thread-mt -lcelero 
SOURCE_DIR=./source_files

CFLAGS=--std=c++11 --default-stream per-thread
MAIN=main_bench.cu

#********************************tests*********************************

#HANDLE_DEPS=$(SOURCE_DIR)/host_location_p.hpp $(SOURCE_DIR)/unified_location_p.cu $(SOURCE_DIR)/pinned_location_p.cu $(SOURCE_DIR)/device_location_p.cpp

#location benchmark
location:location_bench.cu $(MAIN)
	$(CC) -o location $(MAIN) $(FOLDERS) $(LINKS) $(CFLAGS) -D LOCATION_BENCHMARK


#allocation benchmanrk
allocation.bench:allocation.cu 
	python $(UTIL)/coverage.py -i ./allocation.cu
allocation: allocation.cu allocation.bench $(MAIN)
	$(CC) -o allocation $(MAIN) $(FOLDERS) $(LINKS) $(CFLAGS) -D ALLOCATION_BENCHMARK

#all
all:allocation.bench allocation.cu location_bench.cu $(MAIN)
	$(CC) -o benchmark	$(MAIN) $(FOLDERS) $(LINKS) $(CFLAGS) -D ALL

#benchmark
benchmark:all 
	./benchmark -t results.csv

.PHONY:clean format
clean: 
	rm *.bench *.csv -f
format:
	clang-format ./*.cu -style=file -i
	
