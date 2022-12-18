import pandas as pd

log = pd.read_csv('sta3.csv', delimiter=',')
log['timestamp'] = pd.to_datetime(log['timestamp'])

monitor = pd.read_csv('monitor-sw3.csv')
monitor['timestamp'] = pd.to_datetime(monitor['timestamp'])
new_df = log.merge(monitor, on='timestamp', how='inner')

new_df.to_csv('processed_3.csv', index=False)