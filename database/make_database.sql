CREATE TABLE file(
	name		char, --name of the file
	location	char, --directory location of the file
	extension	char,
	PRIMARY KEY(name,location)
	);

CREATE TABLE benchmarkResult(
	bench_group		char,
	ID				integer,
	problem_space		float,
	samples			float,
	iterations		float,
	usperiteration		float,
	iterationspersec	float,	
	min				float,
	mean				float,
	max variance		float,
	skewness			float,
	kurtosis			float,
	PRIMARY KEY(bench_group,ID)
	);

CREATE TABLE bench_param(
	bench_group	char, 
	ID			integer,
	name			char,
	param		char,
	PRIMARY KEY(bench_group,ID,param),
	FOREIGN KEY(bench_group,ID) REFERENCES benchmarkResult(bench_group,ID)
	);


.tables










