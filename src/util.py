from __future__ import division
import psycopg2

def connect_db():
    '''Returns psycopg2 connection to PostgreSQL database.'''
    try:
        conn = psycopg2.connect("dbname='bugs' user='lucka' host='localhost'")
    except:
        print "Unable to connect to the database"
        exit(1)
    return conn