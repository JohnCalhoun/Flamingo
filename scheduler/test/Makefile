CC=nvcc
UTIL=../../utilities
DATAFRAME=../../dataframe/source_files
FOLDERS=-I ../source_files -I ../../allocator/source_files -isystem /usr/include -L/usr/lib64 -I $(UTIL) -I $(DATAFRAME)
LINKS=-lgtest -lpthread -lboost_thread 
SOURCE_DIR=../source_files
COVERAGE=$(UTIL)/coverage.py
CFLAGS=--std=c++11 -g -G -O0 -pg 

#********************************tests*********************************

#task tests
task.test:task.cu $(COVERAGE) ../source_files/*
	python $(COVERAGE) -i ./task.cu
task:task.cu task.test
	clear && $(CC) -o task main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D TASK_TEST

#graph tests
graph.test:graph.cu $(COVERAGE) ../source_files/*
	python $(COVERAGE) -i ./graph.cu
graph:graph.cu graph.test
	clear && $(CC) -o graph main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -ltbb -D GRAPH_TEST

.PHONY:run
run:all
	./test
#clean
.PHONY: clean test
clean:
	rm ./*.test gmon.out -f

test:all
	./test
