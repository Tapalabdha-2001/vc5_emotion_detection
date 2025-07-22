import numpy as np
import pandas as pd
import os
import yaml

from sklearn.model_selection import train_test_split
with open('params.yaml') as f:
    params = yaml.safe_load(f)

test_size = params['data_ingestion']['test_size']


df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
df.drop(columns=['tweet_id'],inplace=True)
final_df = df[df['sentiment'].isin(['happiness','sadness'])]
final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)
# ...existing code...
test_size = float(params['data_ingestion']['test_size'])
# ...existing code...
train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

os.makedirs('data/raw', exist_ok=True)
train_data.to_csv('data/raw/train.csv', index=False)
test_data.to_csv('data/raw/test.csv', index=False)