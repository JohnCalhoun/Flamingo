CC=nvcc
UTIL=../../../utilities
FOLDERS=-I ../source_files -I ../../../allocator/source_files -isystem /usr/include -L/usr/lib64 -I $(UTIL) 
LINKS=-lgtest -lpthread -lboost_thread 
SOURCE_DIR=../source_files
COVERAGE=$(UTIL)/coverage.py
CFLAGS=--std=c++11 -g -G -O0 -pg 

#********************************tests*********************************

#dataframe tests
loadbalancer.test:loadbalancer.cu $(COVERAGE) ../source_files/*
	python $(COVERAGE) -i ./loadbalancer.cu
loadbalancer:loadbalancer.cu loadbalancer.test
	clear && $(CC) -o loadbalancer main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D LOADBALANCER_TEST

.PHONY:run
run:all
	./test
#clean
.PHONY: clean test
clean:
	rm ./*.test gmon.out -f

test:all
	./test
