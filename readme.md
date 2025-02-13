# RL_PPO_TRADING

Reinforcement Learning Trading dengan PPO

1. Candlestick data Bitcoin timeframe 1m
2. Lakukan feature generation via `make_data_v2.py`
3. Data features disimpan di `data/processed.pkl`
4. Jalankan `ppo_trading.py` untuk menjalankan training

# Library

`pip install gymnasium torch pandas numpy`

## **1. Konsep Utama**

### **1.1 Environment**  
Menggunakan data historis sebagai environment untuk simulasi trading.

- **Observation Space:**  
  - `['m5', 'm5_prev', 'm15', 'm15_prev', 'h1', 'h1_prev', 'h4', 'h4_prev','d1', 'd1_prev', 'W1', 'W1_prev', 'M1', 'M1_prev']`
  - Tambahan:
    - `current_mode` → Status posisi (0: Tidak ada posisi, 1: LONG, -1: SHORT)
    - `current_log_return` → Log return dari `open_price` terhadap harga saat ini.

- **Action Space:**  
  `[1, -1, 0]` 
  - `1`: LONG (Buka LONG atau Tutup SHORT)
  - `-1`: SHORT (Buka SHORT atau Tutup LONG)
  - `0`: HOLD (Tidak membuka posisi baru)

- **State Variables:**
  - `open_price`: Harga saat membuka posisi
  - `initial_balance`: Saldo awal
  - `current_balance`: Saldo saat ini
  - `max_balance`: Maksimum saldo yang pernah dicapai (untuk perhitungan drawdown)
  - `current_mode`: Status posisi saat ini

---

### **1.2 Reward Function**  
- **Reward Utama:**  
  ROI saat ini dihitung sebagai:
  \[
  reward = \frac{current\_balance - initial\_balance}{initial\_balance}
  \]

- **Penalty Trading Fee:**  
  Untuk mensimulasikan biaya trading secara realistis, tambahkan penalty trading fee setiap kali terjadi transaksi (buka atau tutup posisi):
  \[
  trading\_fee = abs(traded\_amount) \times fee\_rate
  \]
  Dimana:
    - `traded_amount` = Jumlah aset yang diperdagangkan
    - `fee_rate` = Persentase fee trading (misalnya 0.1% atau 0.001)

  Penalty diberikan dengan mengurangi reward:
  \[
  reward -= trading\_fee
  \]

- **Penalty Tambahan:**
  - **Holding Cost:** Tambahkan holding cost agar agent memahami risiko dari mempertahankan posisi terlalu lama. Misalnya:
    \[
    reward -= 0.00001
    \]
  - **Drawdown Penalty:** Berikan penalti saat drawdown tinggi untuk meminimalkan risiko:
    \[
    drawdown = \frac{max\_balance - current\_balance}{max\_balance}
    \]
    Tambahkan penalti:
    \[
    reward -= drawdown \times penalty\_rate
    \]

- **Reward Tambahan:**
  - Beri insentif untuk menutup posisi yang profit:
    \[
    reward += profit \times profit\_rate
    \]

---

## **2. Arsitektur RL Agent**

### **2.1 Algoritma RL**  
Gunakan **Proximal Policy Optimization (PPO)** untuk kestabilan dan efisiensi dalam pembelajaran. PPO dipilih karena:
- Lebih stabil dibandingkan DQN pada environment yang dinamis seperti market
- Menggunakan clipping pada objective function untuk mencegah update yang terlalu besar

---

### **2.2 Neural Network Architecture**  
- **Input Layer:**
  - Observation Space (`14 + 2 = 16 fitur`)
- **Hidden Layers:**
  - Dua hidden layer dengan **LSTM** untuk menangkap urutan temporal data time series:
    - Layer 1: 64 unit LSTM dengan ReLU Activation
    - Layer 2: 32 unit LSTM dengan ReLU Activation
- **Output Layer:**
  - Softmax untuk probabilitas action (3 output untuk action space `[1, -1, 0]`)

---

## **3. Data Pipeline**  
- **Data Loading:** Gunakan pandas untuk load data historis.
- **Preprocessing:**
  - Normalisasi data pada kolom `['m5', 'm5_prev', ..., 'M1_prev']`
  - Scaling log return untuk menjaga stabilitas training
- **Feature Engineering:**
  - Tambahkan `current_mode` dan `current_log_return`
  - Calculate `current_log_return` sebagai:
    \[
    current\_log\_return = \log\left(\frac{open\_price}{current\_price}\right)
    \]

---

## **4. Training Pipeline**

### **4.1 Training Loop**
1. Initialize environment dan agent
2. **For setiap episode:**
   - Reset environment dan agent state
   - **For setiap timestep:**
     - Ambil observation dan hitung `current_log_return`
     - Tentukan action dengan policy agent
     - Update environment dan state berdasarkan action
     - Hitung reward
     - **Tambahkan penalty trading fee jika terjadi transaksi:**
       - Jika `action` adalah `1` (LONG) atau `-1` (SHORT), hitung `trading_fee` dan kurangi reward
     - Store transition dalam replay buffer
     - Update agent dengan PPO menggunakan mini-batch dari replay buffer
3. **End of Episode:**
   - Update `max_balance`
   - Log performance (ROI, Drawdown, Win Rate)

### **4.2 Early Stopping & Model Checkpoint**
- Gunakan early stopping jika tidak ada peningkatan reward selama N episode
- Simpan checkpoint model terbaik berdasarkan **Sharpe Ratio** dan **Max Drawdown**

---

## **5. Evaluation Metrics**

1. **Cumulative ROI:** Total return dari saldo awal hingga akhir.
2. **Sharpe Ratio:** Mengukur rasio profit terhadap volatilitas.
3. **Max Drawdown:** Risiko maksimum penurunan saldo dari puncak tertinggi.
4. **Win Rate:** Persentase transaksi yang profit.
5. **Holding Period:** Rata-rata waktu holding per trade.
6. **Trading Cost Impact:** Evaluasi pengaruh trading fee terhadap profitabilitas strategi.

---

## **6. Tools dan Frameworks**

- **Gymnasium** untuk environment RL
- **TensorFlow** atau **PyTorch** untuk Neural Network
- **Stable Baselines3** untuk implementasi PPO
- **Pandas** dan **NumPy** untuk data processing
- **Matplotlib** dan **Seaborn** untuk visualisasi performa

---

## **7. Improvement Plan**

1. **Hyperparameter Tuning:**
   - Gunakan **Optuna** atau **Ray Tune** untuk mencari kombinasi hyperparameter terbaik.
2. **Model Ensemble:**
   - Kombinasikan PPO dengan **DQN** atau **A2C** untuk diversifikasi strategi.
3. **Feature Enhancement:**
   - Tambahkan fitur makroekonomi dan on-chain metrics untuk meningkatkan akurasi prediksi.
4. **Transaction Cost Model:**
   - Simulasikan slippage dan biaya trading untuk hasil yang lebih realistis.

---

## **8. Keunggulan Desain Ini**

- **Efisiensi Temporal:** Menggunakan **LSTM** untuk capturing temporal dependencies.
- **Stabilitas Training:** PPO dipilih karena lebih stabil dibandingkan DQN dalam environment finansial.
- **Risiko Terkontrol:** Dengan penalti drawdown, holding cost, dan trading fee, risiko trading bisa lebih terkendali.
- **Realisme Lebih Tinggi:** Trading fee memperhitungkan biaya transaksi untuk hasil yang lebih realistis.
- **Scalable dan Extensible:** Desain ini mudah ditambah dengan fitur atau variabel baru tanpa perubahan besar.

---
