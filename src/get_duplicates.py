from __future__ import division
import requests
import collections
import json
import datetime
from multiprocessing.pool import ThreadPool
from functools import partial
from util import connect_db


requests.adapters.DEFAULT_RETRIES = 2


def get_next_n(conn, limit=0):
    """Gets duplicate ids for n bugs and stores it in PostgreSQL database.

    Args:
        conn: Psycopg2 connection to PostgreSQL database.
        limit (int): Number of bugs to get duplicates for.

    Returns:
        None.
    """
    cur = conn.cursor()
    cur.execute("SET TIME ZONE 'UTC';")

    query = """
            SELECT f.id
            FROM final f
                LEFT OUTER JOIN duplicates d
                    ON f.id = d.id
            WHERE f.resolution_final = 'duplicate'
                AND d.id IS NULL
            LIMIT {}
            """.format(limit)
    cur_report = conn.cursor()
    cur_report.execute(query)

    # use threads to parallelize API requests
    pool = ThreadPool(10)
    bugs = cur_report.fetchall()
    get_bug_desc_conn = partial(get_duplicate_info, conn)
    pool.map(get_bug_desc_conn, bugs)
    pool.close()
    return


def get_duplicate_info(conn, bug):
    """Uses bugzilla API to get bug duplicate id and stores it in PostgreSQL database.

    Args:
        conn: Psycopg2 connection to PostgreSQL database.
        bug (array): Array, the first item is bug_id (integer).

    Returns:
        None. Fetched data is inserted into duplicates table.
    """
    cur = conn.cursor()

    bug_id = str(bug[0])
    # format to https://bugzilla.mozilla.org/rest/bug/505364?include_fields=dupe_of
    website = 'https://bugzilla.mozilla.org/rest/bug/{}?include_fields=dupe_of'.format(bug_id)

    content = requests.get(website)

    # try to parse json data
    try:
        json_content = json.JSONDecoder().decode(content.content)
        duplicate_id = json_content['bugs'][0]['dupe_of']
    except:
        print "Invalid format for bug ", bug_id, ": ", content
        return

    # insert into database
    cur.execute("""
        INSERT INTO duplicates
        VALUES (%s, %s)
        """,
                [bug_id, duplicate_id]
                )
    return

if __name__ == "__main__":
    conn = connect_db()

    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS duplicates (
        id               bigint NOT NULL,
        duplicate_of_id  bigint
        );
    """)
    conn.commit()

    for i in xrange(903):
        get_next_n(conn, limit=100)
        print "{} inserted 100 rows".format(datetime.datetime.now())
        conn.commit()

    conn.close()
