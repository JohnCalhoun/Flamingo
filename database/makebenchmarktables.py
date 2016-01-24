import sqlite3 as db

connection=db.connect("flamingo_database.db")
cursor=connection.cursor()

#get all benchmark types
cursor.execute('''	SELECT DISTINCT bench_group
				FROM benchmarkResult 
				''')
groups=cursor.fetchall()
##get all params for each group
paramaters={}
for group in groups:
	cursor.execute('''	SELECT DISTINCT name
					FROM bench_param
					WHERE bench_group=?''',group)
	paramaters[group]=cursor.fetchall()
##make new tables
for group in paramaters.keys():
	print group
	cursor.execute('''	CREATE TABLE temp (
					ID integer
					)
				''')
	cursor.execute("ALTER TABLE temp RENAME TO %s"%group[0])
	for param in paramaters[group]:
		cursor.execute("ALTER TABLE %s ADD %s char"%(group[0],param[0]) )


















