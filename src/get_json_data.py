from __future__ import division
import psycopg2
import collections
import json
import datetime
from create_tables import create_all_tables
from util import connect_db


def load_table(conn, json_file):
    '''Moves data from json file to PostgreSQL table with the same name.

    Args:
        conn: Psycopg2 connection to PostgreSQL database.
        json_file (str):  Name of json file to be loaded (without extension).

    Returns:
        None.
    '''
    cur = conn.cursor()
    cur.execute("SET TIME ZONE 'UTC';")

    data_path = '/Users/lucka/galvanize/msr2013-bug_dataset/data/v02/mozilla/'
    json_path = data_path + json_file + '.json'
    print '{}: processing {}...'.format(datetime.datetime.now(), json_path)

    with open(json_path, 'r') as f:
        data = json.JSONDecoder(
            object_pairs_hook=collections.OrderedDict).decode(f.read())
        for key, val in data[json_file].iteritems():
            # reports
            if type(val) == collections.OrderedDict:
                query = 'INSERT INTO reports VALUES (to_timestamp(%s), %s, %s, %s, %s)'
                cur.execute(query, val.values() + [int(key)])
            # other tables
            else:
                query = 'INSERT INTO {} VALUES (to_timestamp(%s), %s, %s, %s)'.format(
                    json_file)
                for x in val:
                    cur.execute(query, x.values() + [int(key)])
    conn.commit()
    return

if __name__ == "__main__":

    conn = connect_db()

    tables = [
        'reports',
        'assigned_to',
        'bug_status',
        'cc',
        'component',
        'op_sys',
        'priority',
        'product',
        'resolution',
        'severity',
        'short_desc',
        'version',
    ]
    create_all_tables(conn, tables)

    for table in tables:
        load_table(conn, table)

    conn.close()
