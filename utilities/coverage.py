#make_allocation_benchmark.py
import os 
import re 
import copy
import getopt
import sys
#find functions,types,locations list;
def getoutputfile():
	global contents
	functionRegex=re.compile(r".+python:include=(.*)")
	fun=functionRegex.findall(contents)
	return fun[0]

def getkey():
	global contents
	params={}
	functionRegex=re.compile(r".+python:key:(\w*)=(.*)")
	fun=functionRegex.findall(contents)
	for i in range( len(fun) ):
		params[ fun[i][0] ]=fun[i][1].split(" ")
	return params

def get_template():
	global contents
	functionRegex=re.compile(r".+python:template=(.*)")
	fun=functionRegex.findall(contents)
	return fun

def getnumlines(array,temp):
	num=1
	for i in array:	
		num*=i
	return num*len(temp)

def initializeMaxArray(array):
	global keys
	global paramaters
	for i in range(keys):
		j=paramaters.keys()[i]
		array[i]=len( paramaters[j])

def increment(a,k):
	if k<keys:
		a[k]+=1
		if a[k]>=max_array[k]:
			a[k]=0
			a=increment(a,k+1)
	
# *******************read in file and data
a,b=getopt.getopt(sys.argv[1:],"i:")
input_file=a[0][1]
source_file=open(input_file,"r")
contents=source_file.read()
source_file.close()

#*****************initalize varaibles
paramaters=getkey()
template=get_template()
keys=len( paramaters.keys() )
max_array=[0]*keys
initializeMaxArray(max_array)

lines=getnumlines(max_array,template)
contents_new=[]

for i in range(len(template)):
	contents_new+=[template[i]]*(lines/len(template))
for i in range(len(contents_new)):
	contents_new[i]+='\n'

temps=len(template)
array=[[0]*keys]*(lines*temps)  

for i in range(lines-1):
	array[i+1]=copy.copy(array[i])
	increment(array[i+1],0)

#variable replacement
for j in range(lines):
	for i in range(keys):
		key=paramaters.keys()[i]
		x=array[j][i]
		result=contents_new[j].replace("|"+key+"|",paramaters[key][x])
		contents_new[j]=result
#typedef insertion


typedef_list=[];
typedefreg=re.compile(r".*\$(.+)\$.*")
for k in range(len( contents_new) ):
	matches=typedefreg.findall(contents_new[k] )
	for j in matches:
		match=j
		clear={"<":"_",">":"_",",":"_"," ":""}
	
		for i in clear.keys():
			match= match.replace(i,clear[i] )
	for j in matches:
		typedef=r"typedef "+j+" "+match+"; \n"
		rep="$"+j+"$"
		contents_new[k]=contents_new[k].replace(rep,match)
		typedef_list.append(typedef)

contents_new.insert(0,"//Tests/benchmarks \n")
typedef_list.insert(0,"//typedefs \n")
output=typedef_list+contents_new

outputfile=getoutputfile()

#write out to file
destination_file=open(outputfile,'w')
destination_file.write( "".join(output) )
destination_file.close()











