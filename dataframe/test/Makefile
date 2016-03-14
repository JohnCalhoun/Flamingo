CC=nvcc
UTIL=../../utilities
FOLDERS=-I ../source_files -I ../../allocator/source_files -isystem /usr/include -L/usr/lib64 -I $(UTIL) 
LINKS=-lgtest -lpthread -lboost_thread 
SOURCE_DIR=../source_files
COVERAGE=$(UTIL)/coverage.py
CFLAGS=--std=c++11 -g -G -O0 -pg 

#********************************tests*********************************

#dataframe tests
dataframe.test:container.cu $(COVERAGE) ../source_files/*
	python $(COVERAGE) -i ./container.cu
dataframe_test:container.cu dataframe.test
	clear && $(CC) -o dataframe_test main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D DATAFRAME_TEST

#iterator tests
iterator.test:iterator.cu $(COVERAGE) ../source_files/*
	python $(COVERAGE) -i ./dataframe_test.cu
iterator_test:iterator.cu iterator.test
	clear && $(CC) -o iterator_test main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D ITERATOR_TEST

.PHONY:run
run:all
	./test
#clean
.PHONY: clean test
clean:
	rm ./*.test gmon.out -f

test:all
	./test
