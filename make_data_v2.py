import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

print('Reading CSV...')
df = pd.read_csv('../btcusdt_1m_2017-01-01_2023-06-14.csv', index_col='timestamp', parse_dates=True, usecols=['timestamp', 'open'])

# Filter data mulai dari tanggal yang diinginkan
df = df[df.index >= '2017-12-01']
df['open'] = df['open'].astype(int)

# Resample ke berbagai timeframe dan sinkronkan dengan index df
print('Resampling data...')
timeframes = {
    'm5': '5T',
    'm15': '15T',
    'h1': '1H',
    'h4': '4H',
    'd1': 'D',
    'W1': '1W',
    'M1': 'MS'
}

# Lakukan resample untuk setiap timeframe dan buat kolom _prev setelahnya
for tf, rule in timeframes.items():
    resampled = df['open'].resample(rule).first().ffill()
    df[tf] = resampled.reindex(df.index, method='ffill')
    df[f'{tf}_prev'] = resampled.shift(1).reindex(df.index, method='ffill')

# Filter kembali data mulai tahun 2018
df = df[df.index.year >= 2018]
print(df)

tf_cols = ['m5', 'm5_prev', 'm15', 'm15_prev', 'h1', 'h1_prev', 'h4', 'h4_prev','d1', 'd1_prev', 'W1', 'W1_prev', 'M1', 'M1_prev']
# Tampilkan hasil untuk beberapa baris terakhir
print('\nData with new columns:')
print(df[tf_cols].tail(20))

# Cek apakah ada NaN di kolom baru
print('\nCheck NaN in new columns:')
print(df[tf_cols].isna().sum())



# Tanggal yang ingin dicek
check_date = '2023-06-14'

# List timeframe dan delta-nya
timeframes = {
    'm5': 5,
    'm15': 15,
    'h1': 60,
    'h4': 240,
    'd1': 1440,
    'W1': 10080,  # 7 * 24 * 60
    'M1': 43200  # 30 * 24 * 60
}

# Filter data untuk tanggal yang ingin dicek
df_check = df.loc[check_date]

# Loop untuk setiap timeframe
for tf, delta in timeframes.items():
    # Ambil bar terakhir pada tanggal yang dicek
    last_timestamp = df_check.index[-1]
    # Cari timestamp prev-nya
    prev_timestamp = last_timestamp - pd.Timedelta(minutes=delta)
    
    # Pastikan prev_timestamp ada dalam index dataframe
    if prev_timestamp in df.index:
        # Ambil nilai open di prev_timestamp
        prev_value = df.loc[prev_timestamp, tf]
        # Ambil nilai _prev di last_timestamp
        current_prev_value = df.loc[last_timestamp, f"{tf}_prev"]

        print(f'Memeriksa {tf} pada {last_timestamp} dengan melihat data di {prev_timestamp}')
        if prev_timestamp not in df.index:
            print(f"❓ {tf}_prev tidak bisa diverifikasi, {prev_timestamp} tidak ditemukan di index")

        
        # Bandingkan
        if prev_value == current_prev_value:
            print(f"✅ {tf}_prev benar untuk {last_timestamp}")
        else:
            print(f"⭕ {tf}_prev salah untuk {last_timestamp}: expected {prev_value}, got {current_prev_value}")
    else:
        print(f"🤯 {tf}_prev tidak bisa diverifikasi, timestamp {prev_timestamp} tidak ada dalam data")


print('Menyimpan data raw ke pickle...')
df.to_pickle('data/raw.pkl')

print('Melakukan processing data dengan mengubah tf_cols menjadi log return terhadap harga open ...')
for col in tf_cols:
    df[col] = np.log(df['open'] / df[col])
    # lakukan standarisasi
    df[col] = (df[col] - df[col].mean()) / df[col].std()

print(df)
print(df.info())
print(df.describe())
print('Menyimpan data processed ke pickle...')
df.to_pickle('data/processed.pkl')