CC=nvcc
UTIL=../../utilities
FOLDERS=-I ../source_files -I ../../allocator/source_files -isystem /usr/include -L/usr/lib64 -I $(UTIL) 
LINKS=-lgtest -lpthread -lboost_thread 
SOURCE_DIR=../source_files
COVERAGE=$(UTIL)/coverage.py
CFLAGS=--std=c++11 -g -G -pg -O0

#********************************tests*********************************

#dataframe tests
dataframe.test:dataframe_test.cu $(COVERAGE)
	python $(COVERAGE) -i ./dataframe_test.cu
dataframe_test:dataframe_test.cu dataframe.test
	clear && $(CC) -o dataframe_test main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D DATAFRAME_TEST

.PHONY:run
run:all
	./test
#clean
.PHONY: clean test
clean:
	rm ./*.test gmon.out -f

test:all
	./test
