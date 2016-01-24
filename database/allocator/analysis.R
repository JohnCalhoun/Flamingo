#! /usr/bin/env Rscript
#load libraries
library(DBI)
library(RSQLite)

#set up database connection
drv<-dbDriver("SQLite")
database_location<-"../flamingo_database.db"
db<-dbConnect(drv,database_location)
#loadin data
dbListTables(db)
dbListFields(db,"allocator")
allocator_sql<-"select 'ID' from 'allocator'"
allocator_rs<-dbSendQuery(db,allocator_sql)
allocator_rs

#dbGetRowsAffected(allocator_rs)
#dbColumnInfo(allocator_rs)

#manipulate data

#display data
#close connection
dbDisconnect(db)
