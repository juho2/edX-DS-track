# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 19:43:47 2017

@author: Juho
"""
# Azure DocumentDB

#https://azure.microsoft.com/en-us/resources/videos/create-documentdb-on-azure/
#https://docs.microsoft.com/en-us/azure/documentdb/documentdb-create-account
#https://pypi.python.org/pypi/pydocumentdb/

from pydocumentdb.document_client import DocumentClient

host = 'https://docdb-jke.documents.azure.com:443/'
auth = {'masterKey': '...'}

cli = DocumentClient(host, auth)
db_link = 'dbs/testdb/colls/testcoll'
#doc = {'blah': {'blabla': 'json-mess'}}
#cli.CreateDocument(dblink, doc)
res = cli.QueryDocuments(db_link, 'SELECT * FROM testdb')
res2 = cli.ReadDocuments(db_link)
for doc in res2:
    print(doc['report']['food']['name'])

        