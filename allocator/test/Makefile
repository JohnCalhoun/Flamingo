CC=nvcc
UTIL=../../utilities
FOLDERS=-I ../source_files -isystem /usr/include -L/usr/lib64 -I $(UTIL)
LINKS=-lgtest -lpthread -lboost_thread -ltbb 
SOURCE_DIR=../source_files
COVERAGE=$(UTIL)/coverage.py
CFLAGS=--std=c++11 -g -G -pg -O0 

#********************************tests*********************************

#location tests
location.test:location_test.cu $(COVERAGE)
	python $(COVERAGE) -i ./location_test.cu
location:location_test.cu location.test
	clear && $(CC) -o location_test main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D LOCATION_TEST


#handle tests
HANDLE_DEPS=$(SOURCE_DIR)/Handle.cpp
handle.test:handle_test.cu $(COVERAGE)
	python $(COVERAGE) -i ./handle_test.cu
handle:main_test.cu handle_test.cu handle.test $(HANDLE_DEPS)  
	clear && $(CC) -o handle_test main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D HANDLE_TEST
#handle container test
HANDLE_C_DEPS=$(HANDLE_DEPS) $(SOURCE_DIR)/Handle_Container.cpp
handle_container.test:handle_container_test.cu $(COVERAGE)
	python $(COVERAGE) -i ./handle_container_test.cu
handle_container:main_test.cu handle_container.test $(HANDLE_C_DEPS)
	$(CC) -o handle_container_test main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D HANDLE_C_TEST

#free container test
FREE_CONTAINER_DEPS=$(HANDLE_C_DEPS)  $(SOURCE_DIR)/Free_Container.cpp
free_container.test:free_container_test.cu $(COVERAGE)
	python $(COVERAGE) -i ./free_container_test.cu
free_container:main_test.cu free_container_test.cu free_container.test $(FREE_CONTAINER_DEPS)
	$(CC) -o free_container_test main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D FREE_TEST

#free list test
FREE_LIST_DEPS=$(FREE_CONTAINER_DEPS) $(SOURCE_DIR)/Free_List.cpp
free_list.test:free_list_test.cu $(COVERAGE)
	python $(COVERAGE) -i ./free_list_test.cu
free_list:main_test.cu free_list_test.cu free_list.test $(FREE_LIST_DEPS) 
	$(CC) -o free_list_test main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D FREE_LIST_TEST

#allocation test
ALLOCATION_DEPS=$(FREE_LIST_DEPS) $(SOURCE_DIR)/buddy_alloc_p.cpp $(SOURCE_DIR)/standard_alloc_p.cpp
allocation.test:allocation_test.cu $(COVERAGE)
	python $(COVERAGE) -i ./allocation_test.cu
allocation:main_test.cu allocation_test.cu allocation.test $(ALLOCATION_DEPS)  
	$(CC) -o allocation_test main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D ALLOCATION_TEST

#allocation test
ALLOCATION_DEPS=$(FREE_LIST_DEPS) $(SOURCE_DIR)/buddy_alloc_p.cpp $(SOURCE_DIR)/standard_alloc_p.cpp
allocator.test:allocator_test.cu $(COVERAGE)
	python $(COVERAGE) -i ./allocator_test.cu
allocator:main_test.cu allocator_test.cu allocator.test $(ALLOCATION_DEPS)  
	$(CC) -o allocator_test main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D ALLOCATOR_TEST
#allocato test
ALL_DEPS=$(HANDLE_DEPS) $(SOURCE_DIR)/Reserved_Container.cpp
all:location.test handle.test handle_container.test free_container.test free_list.test allocation.test allocator.test 
	$(CC) -o test  main_test.cu $(FOLDERS) $(LINKS) $(CFLAGS) -D ALL
.PHONY:run
run:all
	./test
#clean
.PHONY: clean test format
clean:
	rm ./*.test gmon.out -f

test:all
	./test

format:
	clang-format ./*_test.* -style=file -i

