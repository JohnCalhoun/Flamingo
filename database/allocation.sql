.mode columns
SELECT ((b.us)/a.us) 
AS ratio ,a.Problem,a.function,a.concurency,a.datatype,a.location 
FROM allocator AS a, allocator AS b 
WHERE a.policy!=b.policy 
	AND a.function=b.function 
	AND a.concurency=b.concurency 
	AND a.datatype=b.datatype 
	AND a.location=b.location 
	AND a.Problem=b.Problem 
	AND a.policy='BUDDY' 
ORDER BY ratio;
