# twitter_topic_mobility_prediction

## Purpose
- Apply Topic Modeling on spatiotemporal tweets
- Train regression model to estimate mobile phone GPS point density from spatiotemporal tweets and topics from the tweets
- Estimate grid-based population density

## Research paper
The paper is going to apper in Frontiers of Computer Science.

    Satoshi Miyazawa, Xuan Song, Tianqi Xia, Ryosuke Shibasaki, and Hodaka Kaneda.
    Integrating GPS trajectory and topics from Twitter stream for human mobility estimation. Frontiers of Computer Science.
    (Accepted: 2017-04-13) DOI: 10.1007/s11704-017-6464-3

PDF is coming soon.

## Files

| File name     | Description                    |
| ------------- | ------------------------------ |
| **data** ||
| places.csv | sample places in Tokyo for model evaluation |
| stoplist_jp.txt | empirical stoplist for Japanese |
| **src** ||
| (config.cfg) | ignored |
| convert_estimation_for_analysis.py | Convert estimation numpy pickle file to CSVs |
| convert_estimation_to_geojson.py | Convert estimation numpy pickle file to GeoJSON |
| convert_gps_to_grid.py | Convert GPS points to density numpy pickle file |
| convert_points_to_grid.py | Define spatiotemporal index for experiment. Slice tweets or gps to create density file |
| convert_topic_feature_for_analysis.py | Convert topic feature numpy pickle file to CSVs |
| convert_topic_feature_to_geojson.py | Convert topic feature numpy pickle file to GeoJSON |
| convert_topic_to_grid | Convert document-topic matrix to numpy pickle file|
| convert_tweet_to_grid_and_gensim.py | Slice tweets and convert to text files for gensim |
| filter_japanese_text.py | Set of Mecab filters |
| filter_text_nightley.py | Filter |
| filter_text_twitter.py | Filter for typical tweet database |
| run_gensim_topicmodels.py | Run gensim topic modeling models to sliced tweets |
| settings.py | Model and experiment settings |
| twitter_topic_mobility_estimation.py | master file to run all process |

## System requirements (docker installation later):
* Python 3.4 or later
* PyMySQL
* MySQL
* config.cfg
* MeCab

You'll need a configuration file with twitter API authentication and MySQL connection information.
As specified on line 18, make a configuration file "config.cfg" in parent directory.
It's a text file. in it, write your twitter API keys and MySQL
connection file like below (replace * with your keys).

```
[twitter]
consumer_key = ****
consumer_secret = ****
access_token_key = ****
access_token_secret = ****

[remote_db]
host = ****
user = ****
passwd = ****
db_name = ****
```

## Data requirements
### Spatiotemporal population density
CSV file of GPS records

### Spatiotemporal tweets
Table of tweets in MySQL database


## Installation (Docker)
```
$ cd twitter-topic-mobility-estimation
$ docker-compose up -d
```

<!--## Testing-->
<!--Run `python setup.py test`-->

## Sample workflow

1. Check and adjust settings.py
    Set model parameters, resource paths.
2. Run filter_text_twitter.py
    This will apply filter for word segmentation.
3. Run twitter_topic_mobility_estimation.py