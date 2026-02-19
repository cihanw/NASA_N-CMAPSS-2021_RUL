import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- ORTAK AYARLAR ---
INPUT_PATH = r'c:\Users\Bilge\OneDrive\Masaüstü\N-CMAPSS RUL\RUL\mamba2_processed.parquet'

# 1. Kodun Ayarları (Health Index)
OUTPUT_DIR_HI = r'c:\Users\Bilge\OneDrive\Masaüstü\N-CMAPSS RUL\RUL\health_index_plots'
SMOOTH_WINDOW = 20

# 2. Kodun Ayarları (Ortalama Sensör Grafikleri)
OUTPUT_DIR_AVG = r'c:\Users\Bilge\OneDrive\Masaüstü\N-CMAPSS RUL\RUL\average_sensor_plots'

def plot_composite_health_index(df):
    """
    1. İŞLEV: Her motor için birleşik sağlık endeksi (Health Index) çizer.
    """
    print("\n--- 1. Health Index Grafikleri Hazırlanıyor ---")
    
    # Klasörü oluştur
    os.makedirs(OUTPUT_DIR_HI, exist_ok=True)
    print(f"Grafikler kaydedilecek yer: {OUTPUT_DIR_HI}")

    # Verinin kopyası üzerinde çalışalım ki diğer fonksiyonu etkilemesin
    df_hi = df.copy()

    # Tüm Residual (R_) sütunlarını al
    res_cols = [c for c in df_hi.columns if c.startswith('R_')]
    print(f"HI Hesabı için Kullanılan Sensör Sayısı: {len(res_cols)}")
    
    # --- HEALTH INDEX HESAPLAMA ---
    # Yöntem: RMS (Root Mean Square) benzeri yaklaşım
    df_hi['Health_Index'] = np.sqrt((df_hi[res_cols] ** 2).sum(axis=1))
    
    # Örnek olarak ilk 3 üniteyi çizelim (İsteğe göre değiştirilebilir)
    target_units = df_hi['unit'].unique()[:3] 
    
    for unit_id in target_units:
        print(f"Unit {unit_id} için Health Index çiziliyor...")
        
        unit_df = df_hi[df_hi['unit'] == unit_id].sort_values('cycle')
        
        plt.figure(figsize=(12, 6))
        
        # 1. Ham Health Index (Gri)
        plt.plot(unit_df['cycle'], unit_df['Health_Index'], 
                 color='gray', alpha=0.3, label='Anlık HI (Raw)')
        
        # 2. Yumuşatılmış Trend (Kırmızı)
        hi_smooth = unit_df['Health_Index'].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
        
        plt.plot(unit_df['cycle'], hi_smooth, 
                 color='#d62728', linewidth=3, label=f'Health Trend (Smoothed {SMOOTH_WINDOW})')
        
        # Eşik Değeri Çizgisi
        plt.axhline(y=hi_smooth.iloc[0]*2, color='orange', linestyle='--', label='Potansiyel Alarm Seviyesi')

        plt.title(f'Unit {unit_id}: Composite Health Index (Tüm Sensörlerin Birleşimi)', fontsize=14)
        plt.xlabel('Cycle', fontsize=12)
        plt.ylabel('Health Index (Bozulma Büyüklüğü)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        filename = f"Health_Index_Unit_{unit_id}.png"
        plt.savefig(os.path.join(OUTPUT_DIR_HI, filename), dpi=150)
        plt.close()

    print("Health Index işlemi tamamlandı.")

def plot_population_average(df):
    """
    2. İŞLEV: Tüm filonun ortalama sensör hatasını ve standart sapmasını çizer.
    """
    print("\n--- 2. Filonun Ortalama Sensör Grafikleri Hazırlanıyor ---")
    
    # Klasörü oluştur
    os.makedirs(OUTPUT_DIR_AVG, exist_ok=True)
    print(f"Grafikler kaydedilecek yer: {OUTPUT_DIR_AVG}")

    # Residual (Hata) sütunlarını bul
    res_cols = [c for c in df.columns if c.startswith('R_')]
    
    print("Tüm üniteler üzerinden ortalamalar hesaplanıyor...")
    
    # İstatistikleri Hesapla: Cycle bazında ortalama ve standart sapma
    stats_df = df.groupby('cycle')[res_cols].agg(['mean', 'std'])
    
    # Çizim Döngüsü
    for col in res_cols:
        # print(f"Çiziliyor: {col}...") # Çok kalabalık olmasın diye kapattım, açabilirsiniz.
        
        plt.figure(figsize=(10, 6))
        
        # Verileri çek
        cycles = stats_df.index
        mean_val = stats_df[col]['mean']
        std_val = stats_df[col]['std']
        
        # A) Ortalama Çizgisi (Koyu Mavi)
        plt.plot(cycles, mean_val, 
                 label='Filo Ortalaması (Mean)', 
                 color='#005b96', 
                 linewidth=2)
        
        # B) Standart Sapma Bandı (Açık Mavi Gölge)
        plt.fill_between(cycles, 
                         mean_val - std_val, 
                         mean_val + std_val, 
                         color='#b3cde0', 
                         alpha=0.5, 
                         label='Standart Sapma (±1 Std)')
        
        # Grafik Süslemeleri
        sensor_name = col.replace('R_', '')
        plt.title(f'{sensor_name} Sensörü: Tüm Filonun Ortalama Hata Eğrisi', fontsize=14)
        plt.xlabel('Cycle (Uçuş Döngüsü)', fontsize=12)
        plt.ylabel('Ortalama Hata (Mean Residual)', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Kaydet
        filename = f"Avg_Trend_{sensor_name}.png"
        plt.savefig(os.path.join(OUTPUT_DIR_AVG, filename), dpi=150)
        plt.close()

    print("Ortalama grafik işlemi tamamlandı.")

def main():
    # 1. Veriyi Tek Seferde Yükle
    print(f"Veri yükleniyor: {INPUT_PATH} ...")
    if not os.path.exists(INPUT_PATH):
        print("HATA: Dosya bulunamadı!")
        return
        
    df = pd.read_parquet(INPUT_PATH)
    print(f"Veri yüklendi. Boyut: {df.shape}")
    
    # 2. Fonksiyonları Sırayla Çalıştır
    plot_composite_health_index(df)
    plot_population_average(df)
    
    print("\n*** TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI ***")

if __name__ == "__main__":
    main()