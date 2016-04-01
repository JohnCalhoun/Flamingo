TEST=./test
SOURCE=./source_files
BENCH=./benchmarks
#format 
.PHONY:test-format benchmark-format source-format format
test-format:
	make format -C $(TEST)
benchmark-format:
	make format -C $(BENCH)
source-format:
	make format -C $(SOURCE)
format:test-format benchmark-format source-format
#clean
.PHONY:test-clean benchmark-clean source-clean clean
test-clean:
	make clean -C $(TEST) 
benchmark-clean:
	make clean -C $(BENCH)
source-clean:
	make clean -C $(SOURCE)
clean:test-clean benchmark-clean source-clean
#build 

#test
.PHONY:test
run: 
	make run -F $(TEST)

#commit
.PHONY:phase_one
phase_one:format test

.PHONY:commit
commit:
	git commit -a -F commit.txt

.PHONY:push
push:
	git push origin master
