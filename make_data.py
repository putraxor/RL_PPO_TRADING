import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

print('Reading CSV...')
df = pd.read_csv('../btcusdt_1m_2017-01-01_2023-06-14.csv', index_col='timestamp', parse_dates=True, usecols=['timestamp', 'open'])
#ambil data yang diatas 2017-08-18
df['open'] = df['open'].astype(np.float32)
df = df[df.index >= '2017-08-18']


print('TF 1m')
print(df)

print('\n\nResampling to 5m...')
m5 = df.resample('5T').first().ffill()
print(m5)
print('Baris dengan NaN:', m5.isna().sum())

print('\n\nResampling to 15m...')
m15 = df.resample('15T').first().ffill()
print(m15)
print('Baris dengan NaN:', m15.isna().sum())

print('\n\nResampling to 1h...')
h1 = df.resample('1H').first().ffill()
print(h1)
print('Baris dengan NaN:', h1.isna().sum())


print('\n\nResampling to 4h...')
h4 = df.resample('4H').first().ffill()
print(h4)
print('Baris dengan NaN:', h4.isna().sum())

print('\n\nResampling to 1d...')
d1 = df.resample('D').first().ffill()
print(d1)
print('Baris dengan NaN:', d1.isna().sum())

print('\n\nResampling to 1W...')
W1 = df.resample('1W').first().ffill()
print(W1)
print('Baris dengan NaN:', W1.isna().sum())

print('\n\nResampling to 1M...')
M1 = df.resample('MS').first().ffill()
print(M1)
print('Baris dengan NaN:', M1.isna().sum())

# pembuatan kolom ['m5', 'm5_prev', 'm15', 'm15_prev', 'h1', 'h1_prev', 'h4', 'h4_prev', 'd1', 'd1_prev', 'W1', 'W1_prev', 'M1', 'M1_prev']