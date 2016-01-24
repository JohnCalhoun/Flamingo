import os
import sqlite3 as db
rootDir="/home/john/projects/flamingo/"
Directories=['allocator','container']
database="flamingo_database.db"

files=[]
def getfiles(directory):
	for dirName, subdirList, fileList in os.walk(directory):
		for fname in fileList:
			if not ('.git' in dirName):
				files.append([dirName,fname])			

for dir in Directories:
	getfiles(rootDir+dir)
data=[]
for i in files:
	location=[i[0]]
	file=[i[1]]
	extension=[i[1].split(".")[-1]]
	data.append(location+file+extension)

connection=db.connect(database)
cur=connection.cursor()

cur.executemany('INSERT INTO file VALUES(?,?,?)',data)
connection.commit()
connection.close()
