# -*- coding: utf-8 -*-

# http://www.python-course.eu/sql_python.php
# http://pythoncentral.io/introductory-tutorial-python-sqlalchemy/

## SQL server on Azure, Transact-SQL

##MS tutorial
#import pyodbc
#server = 'test-jke.database.windows.net'
#database = 'sql-jke'
#username = 'sql-jke'
#password = '...'
#driver= '{ODBC Driver 13 for SQL Server}'
#cnxn = pyodbc.connect('DRIVER='+driver+';PORT=1433;SERVER='+server+';PORT=1443;DATABASE='+database+';UID='+username+';PWD='+ password)
#cursor = cnxn.cursor()
#cursor.execute("select @@VERSION")
#row = cursor.fetchone()
#while row:
#    print(str(row[0]))# + " " + str(row[1])) + " " + str(row[2]))
#    row = cursor.fetchone()
#
#cursor.execute("SELECT * FROM SalesLT.Product p")
#row = cursor.fetchone(); print(row)

from sqlalchemy import create_engine, inspect
engine = create_engine("mssql+pyodbc://sql-jke:<PWD>@test-jke.database.windows.net:1433/sql-jke?driver=ODBC Driver 13 for SQL Server")
insp = inspect(engine)
print(insp.get_schema_names())
print(engine.table_names(schema='saleslt'))

conn = engine.connect()
import pandas as pd
customers = pd.read_sql_table('Customer', conn, schema='saleslt')
orders = pd.read_sql_table('SalesOrderHeader', conn, schema='saleslt')

query1 = 'SELECT CustomerID AS id, CONCAT(FirstName, LastName) AS Name, ModifiedDate as Date \
         FROM saleslt.Customer'
res1 = pd.read_sql(query1, conn)
print(res1.head(), '\n', res1.shape)

query2 = " \
        SELECT OrderDate, COUNT(SalesOrderNumber) \
        FROM saleslt.SalesOrderHeader \
        WHERE TotalDue > 1000 \
        GROUP BY OrderDate \
        HAVING COUNT(SalesOrderNumber) > 0 \
        "
res2 = pd.read_sql(query2, conn)
print(res2.head(), '\n', res2.shape)

query3 = " \
        SELECT SalesOrderID, TotalDue, (TotalDue-SubTotal)/SubTotal AS ExpensePct \
        FROM saleslt.SalesOrderHeader \
        --WHERE TotalDue > 1000 \
        ORDER BY TotalDue DESC; \
        "
res3 = pd.read_sql(query3, conn)
print(res3.head(), '\n', res3.shape)

query4 = " \
        SELECT ord.CustomerID, CONCAT(cus.FirstName, cus.LastName), ord.TotalDue \
        FROM saleslt.SalesOrderHeader AS ord \
        INNER JOIN saleslt.Customer AS cus \
        ON ord.CustomerID = cus.CustomerID \
        "
res4 = pd.read_sql(query4, conn)
print(res4, '\n', res4.shape)