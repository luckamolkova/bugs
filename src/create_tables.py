def create_all_tables(conn, tables):
    '''Creates empty tables unless they already exist.

    Args:
        conn: Psycopg2 connection to PostgreSQL database.
        tables (list): List of tables to be created in addition to reports table.

    Returns:
        None.
    '''
    cur = conn.cursor()

    # create reports table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
        opening             timestamp,
        reporter            bigint,
        current_status      varchar(50),
        current_resolution  varchar(50),
        id                  bigint NOT NULL
        );
    """)
    conn.commit()

    # create other tables
    for table in tables:
        query = """
            CREATE TABLE IF NOT EXISTS {} (
            when_created    timestamp,
            what            text,
            who             bigint,
            id              bigint NOT NULL
            );
        """.format(table)
        cur.execute(query)
        conn.commit()

    return
