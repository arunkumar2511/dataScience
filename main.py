import psycopg2

CONNECTION = "postgres://tsdbadmin:pgziyphxveb9rklk@wermaj7i3e.tml67azc7u.tsdb.cloud.timescale.com:32395/tsdb?sslmode=require"
def main():  
    conn = psycopg2.connect(CONNECTION)
    cursor = conn.cursor()
    # use the cursor to interact with your database
    cursor.execute("SELECT 'hello world'")
    print(cursor.fetchone())

main()