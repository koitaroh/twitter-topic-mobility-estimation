# Last Update: 2016-07-27
# @author: Satoshi Miyazawa
# koitaroh@gmail.com
# Specify a table on a database. Apply filter_japanese_text.py on corresponding column.

import sqlalchemy
from filter_japanese_text import filter_japanese_text
import settings as s

# Logging ver. 2016-07-12
from logging import handlers
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler('log.log', maxBytes=1000000, backupCount=3)  # file handler
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()  # console handler
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('Initializing %s', __name__)


def prepare_tweet_nightley_db(engine_conf, in_table):
    engine = sqlalchemy.create_engine(engine_conf, echo=False)
    conn = engine.connect()
    metadata = sqlalchemy.MetaData(engine)
    table = sqlalchemy.Table(in_table, metadata, autoload=True, autoload_with=engine)
    s = sqlalchemy.select([table.c.id_str, table.c.text]).where(table.c.text != None)
    result = conn.execute(s)
    for row in result.fetchall():
        id_str = row['id_str']
        text = row['text']
        words = filter_japanese_text(text)
        logger.debug(words)
        stmt = table.update().where(table.c.id_str == id_str).values(words=words)
        conn.execute(stmt)
    result.close()
    return True

if __name__ == '__main__':
    engine_conf = s.ENGINE_CONF
    table_name = s.TABLE_NAME
    test1 = "帰るよー (@ 渋谷駅 (Shibuya Sta.) in 渋谷区, 東京都) https://t.co/UwXP9Gr0wJ check this out (http://t.co/nYHbleBtBT)"
    test2 = "終電変わらず(｀・ω・´)ゞ @ 川崎駅にタッチ！ http://t.co/DJFKEEUW3n"
    test3 = "月ザンギョ100とか200とかそら死ぬわな、と実感しつつある。 (@ ノロワレハウス II in 杉並区, 東京都) https://t.co/BkcI7uNigi"
    # prepare_tweet_nightley_db(engine_conf, "social_activity_201307_test")
    prepare_tweet_nightley_db(engine_conf, table_name)


