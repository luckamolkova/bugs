from __future__ import division
import psycopg2
import datetime
from util import connect_db


def create_base_table(conn, base_table='base'):
    '''Creates base table.

    Args:
        conn: Psycopg2 connection to PostgreSQL database.
        base_table (string): Name of the base table, defaults to `base`.

    Returns:
        None. Table named `base` is created in database.
    '''
    print '{}: creating {} table...'.format(datetime.datetime.now(), base_table)
    cur = conn.cursor()
    query = 'DROP TABLE IF EXISTS {};'.format(base_table)
    cur.execute(query)
    query = """
        CREATE TABLE base AS (
            SELECT
                r.id
                , r.opening
                , r.reporter
                , r.current_status
                , r.current_resolution
            FROM reports r
                LEFT OUTER JOIN descriptions d
                    ON r.id = d.id
            WHERE 1 = 1
                AND (d.when_created - r.opening) < interval '10 m'
                AND r.current_status in ('RESOLVED', 'VERIFIED', 'CLOSED')
        );
    """
    cur.execute(query)
    conn.commit()
    return


def create_clean_table(conn, table):
    '''Creates cleaned up table.

    Args:
        conn: Psycopg2 connection to PostgreSQL database.
        table (str):  Name of table.

    Returns:
        None.
    '''
    print '{}: creating {}_clean table...'.format(datetime.datetime.now(), table)
    cur = conn.cursor()
    query = 'DROP TABLE IF EXISTS {}_clean;'.format(table)
    cur.execute(query)
    query = """
        CREATE TABLE {}_clean AS (
            SELECT
                CASE
                    WHEN what IS NULL THEN ''
                    ELSE LOWER(TRIM(what))
                END as what
                , when_created
                , id
            FROM {}
        );
    """.format(table, table)
    cur.execute(query)
    conn.commit()
    return


def create_temp_table(conn, table):
    '''Creates temporary table.

    Args:
        conn: Psycopg2 connection to PostgreSQL database.
        table (str):  Name of table.

    Returns:
        None.
    '''
    print '{}: creating {}_2 temp table...'.format(datetime.datetime.now(), table)
    cur = conn.cursor()
    query = 'DROP TABLE IF EXISTS {}_2;'.format(table)
    cur.execute(query)
    query = """
        CREATE TABLE {}_2 AS (
            WITH helper AS (
                SELECT
                    t.*
                    , rank() OVER (PARTITION BY t.id ORDER BY t.when_created ASC) AS when_first
                    , rank() OVER (PARTITION BY t.id ORDER BY t.when_created DESC) AS when_last
                FROM {}_clean t
            )
            SELECT
                b.id
                , COUNT(h.id) AS count_val
                , COALESCE(MAX(h.what) FILTER (WHERE h.when_first = 1 AND b.opening = h.when_created), '') AS initial_val
                , COALESCE(MAX(h.what) FILTER (WHERE h.when_last = 1), '') AS final_val
                , COALESCE(MAX(h.when_created) FILTER (WHERE h.when_last = 1), NULL) AS final_when
            FROM base b
                LEFT JOIN helper h
                    ON b.id = h.id
            GROUP BY b.id
        );
    """.format(table, table)
    cur.execute(query)
    conn.commit()
    return


def create_reorter_bugs_table():
    # ```
    # CREATE TABLE IF NOT EXISTS reporter_bugs AS (
    #   SELECT
    #     f1.id AS id
    #     , COUNT(f2.id) AS reporter_bug_cnt
    #   FROM final f1
    #     LEFT OUTER JOIN final f2
    #       ON f1.reporter = f2.reporter
    #       AND f1.opening > f2.opening
    #   GROUP BY f1.id
    # );
    # ```
    pass


def create_final_table(conn, base_table='base', final_table='final'):
    '''Creates final table.

    Args:
        conn: Psycopg2 connection to PostgreSQL database.
        base_table (string): Name of the base table, defaults to `base`.
        final_table (string): Name of the final table, defaults to `final`.

    Returns:
        None. Table named `final` is created in database.
    '''
    create_reorter_bugs_table()

    print '{}: creating {} table...'.format(datetime.datetime.now(), final_table)
    cur = conn.cursor()
    query = 'DROP TABLE IF EXISTS {};'.format(final_table)
    cur.execute(query)
    query = """
        CREATE TABLE {} AS (
                SELECT
                    b.id AS id
                    , b.opening
                    , b.reporter
                    , LOWER(TRIM(b.current_status)) AS current_status
                    , LOWER(TRIM(b.current_resolution)) AS current_resolution
                    , a.count_val AS assigned_to_cnt
                    , a.initial_val AS assigned_to_init
                    , a.final_val AS assigned_to_final
                    , bg.count_val AS bug_status_cnt
                    , bg.initial_val AS bug_status_init
                    , bg.final_val AS bug_status_final
                    , cc.count_val AS cc_cnt
                    , cc.initial_val AS cc_init
                    , cc.final_val AS cc_final
                    , c.count_val AS component_cnt
                    , c.initial_val AS component_init
                    , c.final_val AS component_final
                    , o.count_val AS op_sys_cnt
                    , o.initial_val AS op_sys_init
                    , o.final_val AS op_sys_final
                    , p.count_val AS priority_cnt
                    , p.initial_val AS priority_init
                    , p.final_val AS priority_final
                    , pr.count_val AS product_cnt
                    , pr.initial_val AS product_init
                    , pr.final_val AS product_final
                    , r.count_val AS resolution_cnt
                    , r.initial_val AS resolution_init
                    , r.final_val AS resolution_final
                    , s.count_val AS severity_cnt
                    , s.initial_val AS severity_init
                    , s.final_val AS severity_final
                    , sh.count_val AS short_desc_cnt
                    , sh.initial_val AS short_desc_init
                    , sh.final_val AS short_desc_final
                    , v.count_val AS version_cnt
                    , v.initial_val AS version_init
                    , v.final_val AS version_final
                    , d.descr AS desc_init
                    , r.final_when AS closing
                    , rb.reporter_bug_cnt AS reporter_bug_cnt
                FROM {} b
                    JOIN assigned_to_2 a ON b.id = a.id
                    JOIN bug_status_2 bg ON b.id = bg.id
                    JOIN cc_2 cc ON b.id = cc.id
                    JOIN component_2 c ON b.id = c.id
                    JOIN op_sys_2 o ON b.id = o.id
                    JOIN priority_2 p ON b.id = p.id
                    JOIN product_2 pr ON b.id = pr.id
                    JOIN resolution_2 r ON b.id = r.id
                    JOIN severity_2 s ON b.id = s.id
                    JOIN short_desc_2 sh ON b.id = sh.id
                    JOIN version_2 v ON b.id = v.id
                    JOIN descriptions d ON b.id = d.id
                    JOIN reporter_bugs rb ON b.id = rb.id
        );
    """.format(final_table, base_table)
    cur.execute(query)
    conn.commit()
    return

if __name__ == "__main__":

    conn = connect_db()

    tables = ['assigned_to',
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

    create_base_table(conn)

    # for table in tables:
    #     create_clean_table(conn, table)
    #     create_temp_table(conn, table)

    create_final_table(conn)

    conn.close()
