import pandas as pd

log = pd.read_csv('player-sta1.csv', delimiter=',')
log['timestamp'] = pd.to_datetime(log['timestamp'])

monitor = pd.read_csv('monitor-sw1.csv')
monitor['timestamp'] = pd.to_datetime(monitor['timestamp'])
new_df = log.merge(monitor, on='timestamp', how='inner')

new_df.to_csv('processed_1.csv', index=False)