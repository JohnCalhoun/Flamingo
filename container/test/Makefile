CC=nvcc
UTIL=/home/john/projects/flamingo/utilities
LINKS=-lgtest -lpthread -lboost_thread-mt 
SOURCE_DIR=/home/john/projects/flamingo/container/source_files
COVERAGE=$(UTIL)/coverage.py
CFLAGS=--std=c++11 -g -G -pg -O0 
FOLDERS=-I /home/john/projects/flamingo/allocator/source_files -isystem /usr/include -L/usr/lib64 -I $(SOURCE_DIR) -I $(UTIL)

#********************************tests*********************************
.PHONY:clear_screen

clear_screen:
	clear

#root tests
root.test:root_test.cu $(COVERAGE)
	python $(COVERAGE) -i ./root_test.cu
root:root_test.cu root.test clear_screen
	$(CC) -o root_test main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D ROOT_TEST
rootRun:root
	./root_test
#HashedArrayTree tests
tree.test:tree_test.cu $(COVERAGE)
	python $(COVERAGE) -i ./tree_test.cu
tree:tree_test.cu tree.test clear_screen
	$(CC) -o tree_test main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D TREE_TEST

#utility tests
utility.test:utility_test.cu $(COVERAGE)
	python $(COVERAGE) -i ./utility_test.cu
utility:utility_test.cu utility.test clear_screen
	$(CC) -o utility_test main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D UTILITY_TEST

#HashedArrayTree tests
hashedarraytree.test:hashedarraytree_test.cu $(COVERAGE)
	python $(COVERAGE) -i ./hashedarraytree_test.cu
container:hashedarraytree_test.cu hashedarraytree.test clear_screen
	$(CC) -o hashedarraytree_test main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D HASHEDARRAYTREE_TEST


#iterator tests
iterator.test:iterator_test.cu $(COVERAGE)
	python $(COVERAGE) -i ./iterator_test.cu
iterator:iterator_test.cu iterator.test clear_screen
	$(CC) -o iterator_test main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D ITERATOR_TEST



#all
all:hashedarraytree.test clear_screen
	$(CC) -o test  main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D ALL

#clean
.PHONY: clean test
clean:
	rm ./*.test gmon.out

test:all
	./test

