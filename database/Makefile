BASEDIR=~/projects/flamingo/database
DATABASE=$(BASEDIR)/flamingo_database.db

.PHONY:clean
.PHONY:getfiles getbenchmarks benchmarktables
.PHONY:remake-clean remake-database remake-input remake
clean:
	rm $(DATABASE)
getfiles:
	python get_files.py
getbenchmarks:
	python get_benchmarks.py
benchmarktables:
	python makebenchmarktables.py


database: $(BASEDIR)/make_database.sql clean
	sqlite3 $(DATABASE) < $(BASEDIR)/make_database.sql


remake-clean:clean
remake-database:database remake-clean
remake-input:remake-database getfiles getbenchmarks

remake:remake-input


