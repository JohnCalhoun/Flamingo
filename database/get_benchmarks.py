import os
import sqlite3 as db
import csv

#setup connection
connection=db.connect('flamingo_database.db')
cursor=connection.cursor()
#get csv file
cursor.execute("""	SELECT * 
				FROM file 
				WHERE extension='csv'
			""")
resultfiles=cursor.fetchall()
#get template file
cursor.execute("""	SELECT * 
				FROM file 
				WHERE name='benchmark.config'
			""")
benchmarkconfigfile=cursor.fetchall()
#get results
results=[]
for i in resultfiles:
	file=i[0]+'/'+i[1]
	csvfile=open(file,"rb")
	reader=list(csv.reader(csvfile) )
	results=results+reader

resultoutput=[]
paramoutput=[]
rows=len(results)


header=results[0];
header=[x.split(" ")[0] for x in header]
header=[x.split("/")[0] for x in header]
header_types=[" FLOAT"]*len(header)
zippedheader=zip(header,header_types)
header=["".join(x) for x in zippedheader]

##get group names
groups={}
for j in range(rows-1):
	i=j+1
	group=(results[i][0].split("_")[0])
	names=results[i][1].split("_")[::3]	
	groups[group]=names
#create tables for each group
tablecreations=[]
for group in groups.keys():
	tables=",".join(header[2:4]+header[6:-2]+groups[group])
	tablecreations.append("".join(["CREATE TABLE ",group,'(','ID INTEGER,']+[tables]+[',PRIMARY KEY (ID)',')']))
for creation in tablecreations:
	cursor.execute(creation)

#in put data into appropriate table
for j in range(rows-1):
	i=j+1
	#parse results
	group=(results[i][0].split("_")[0])
	outputs=results[i][2:4]+results[i][6:-2]
	outputs=[float(x) for x in outputs]
	param=results[i][1].split("__")
	param=[x.split('_')[1] for x in param]
	output=([i]+outputs+param)
	#insert in database
	insert="INSERT INTO %s VALUES"%group
	questionmarks="".join(["(",",".join(["?"]*len(output)),")"])
	cursor.execute(insert+questionmarks,output)

connection.commit()
connection.close()






