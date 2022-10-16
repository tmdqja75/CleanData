import pandas as pd
import sqlalchemy.dialects.mysql.types
from sqlalchemy import create_engine

class Conn:
    def __init__(self, id, pwd, host='localhost', name='catchvdatabase'):
        self.db_cfg = dict()
        self.db_cfg['db_id'] = id
        self.db_cfg['db_pwd'] = pwd
        self.db_cfg['db_host'] = host
        self.db_cfg['db_name'] = name
        self.conn = self.conn_db()

    def conn_db(self):
        db_conn_str = f"mysql+pymysql://{self.db_cfg['db_id']}:" \
                      f"{self.db_cfg['db_pwd']}@" \
                      f"{self.db_cfg['db_host']}/" \
                      f"{self.db_cfg['db_name']}"
        db_conn = create_engine(db_conn_str)
        conn = db_conn.connect()
        return conn

    def df2facedata(self, dataframe):
        dtypesql={'id':sqlalchemy.types.INT,
                  'image': sqlalchemy.dialects.mysql.types.MEDIUMBLOB,
                   'name': sqlalchemy.types.VARCHAR(100),
                   'size': sqlalchemy.dialects.mysql.MEDIUMTEXT,
                   'uploader': sqlalchemy.types.VARCHAR(40),
                   'startDate': sqlalchemy.types.VARCHAR(100)}
        try:
            dataframe.to_sql(name='facedata', con=self.conn, if_exists='append',
                          index=False, dtype=dtypesql)
            return True
        except:
            return False

    def df2resultdata(self, dataframe):
        dtypesql = {'id': sqlalchemy.types.INT,
                    'videoCount': sqlalchemy.types.INT,
                    'detectCount': sqlalchemy.types.INT,
                    'userEmail': sqlalchemy.types.VARCHAR(40),
                    'urlList': sqlalchemy.types.VARCHAR(200)}
        try:
            dataframe.to_sql(name='resultdata', con=self.conn, if_exists='append',
                             index=False, dtype=dtypesql)
            return True
        except:
            return False

    def df2user(self, dataframe):
        dtypesql = {'id': sqlalchemy.types.INT,
                    'userEmail': sqlalchemy.types.CHAR(40),
                    'userPassword': sqlalchemy.types.CHAR(100),
                    'loginstatus': sqlalchemy.dialects.mysql.TINYINT}
        try:
            dataframe.to_sql(name='user', con=self.conn, if_exists='append',
                             index=False, dtype=dtypesql)
            return True
        except:
            return False

    def read_table(self, table_name):
        sql = f"select * from {table_name}"
        result = pd.read_sql_query(sql, self.conn)
        return result

    def conn_close(self):
        self.conn.close()