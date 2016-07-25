from __future__ import division
import requests
import collections
import json
import psycopg2
import time
import datetime
from multiprocessing.pool import ThreadPool
from functools import partial
from util import connect_db


requests.adapters.DEFAULT_RETRIES = 2


def get_next_n(conn, limit=0):
    ''' Gets description (first comment) for n bugs and stores it in PostgreSQL database.

    Args:
        conn: Psycopg2 connection to PostgreSQL database.
        limit (int): Number of bugs to get description for.

    Returns:
        None.
    '''
    cur = conn.cursor()
    cur.execute("SET TIME ZONE 'UTC';")

    query = """
            SELECT r.id
            FROM reports r
                LEFT OUTER JOIN descriptions d
                    ON r.id = d.id
            WHERE d.id IS NULL
                -- exclude bugs that are not publicly available
                AND r.id NOT IN (516716, 802791, 808848, 808853,
                                821596, 627634, 525063, 661158,
                                758675, 773487, 808808, 804674,
                                724355, 795589, 808857, 574459)
            LIMIT {}
            """.format(limit)
    cur_report = conn.cursor()
    cur_report.execute(query)

    # use threads to parallelize API requests
    pool = ThreadPool(10)
    bugs = cur_report.fetchall()
    get_bug_desc_conn = partial(get_bug_desc, conn)
    pool.map(get_bug_desc_conn, bugs)
    pool.close()
    return


def get_bug_desc(conn, bug):
    ''' Uses bugzilla API to get first bug comment and stores it in PostgreSQL database.

    Args:
        conn: Psycopg2 connection to PostgreSQL database.
        bug (array): Array, the first item is bug_id (integer).

    Returns:
        None. Fetched data is inserted into descriptions table.
    '''
    cur = conn.cursor()

    bug_id = str(bug[0])
    # format to https://bugzilla.mozilla.org/rest/bug/707428/comment
    website = 'https://bugzilla.mozilla.org/rest/bug/{}/comment'.format(bug_id)

    content = requests.get(website)

    # try to parse json data
    try:
        json_content = json.JSONDecoder().decode(content.content)
        bug_date = json_content['bugs'][bug_id]['comments'][0]['creation_time']
        bug_desc = json_content['bugs'][bug_id]['comments'][0]['raw_text']
    except:
        print "Invalid format for bug ", bug_id, ": ", content
        return

    # insert into database
    cur.execute("""
        INSERT INTO descriptions
        VALUES (%s, to_timestamp(%s, 'YYYY-MM-DD HH24:MI:SS') - interval '9 h', %s)
        """,
        [bug_id, bug_date, bug_desc]
    )
    return

if __name__ == "__main__":
    conn = connect_db()

    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS descriptions (
        id            bigint NOT NULL,
        when_created  timestamp,
        descr         text
        );
    """)
    conn.commit()

    for i in xrange(1000):
        get_next_n(conn, limit=100)
        print "{} inserted 100 rows".format(datetime.datetime.now())
        conn.commit()

    conn.close()
