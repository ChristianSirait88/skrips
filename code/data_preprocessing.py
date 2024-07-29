import pandas as pd
import numpy as np

# Read data from an Excel file 
df = pd.read_excel('Data\Data Valuasi Polis Asuransi Kesehatan BNI Life - Data Mahasiswa atn. Christian.xlsx')

# ----- Start Missing Value Imputation -------

# Remove rows with '0' values in 'BAYAR' column
df = df[df['BAYAR'] != 0]

# Remove rows with null values in the "JENISKELAMIN" column
df = df[df['JENISKELAMIN'] != 0]

# Remove rows with null values in the "NAMAPENYAKIT" column
df = df[df['NAMAPENYAKIT'] !="-"]
df = df.dropna(subset=['NAMAPENYAKIT'])

# Remove rows with null values in the "JENISKLAIM" column
df = df.dropna(subset=['JENISKLAIM'])

df = df.dropna(subset=['BAYAR'])

# Remove rows with null values in the "KARYAWAN" column
df = df.dropna(subset=['KARYAWAN'])

# Remove the "PERIODE" column
df.drop(columns=['PERIODE'], inplace=True)

# Remove the "NAMAPESERTA" column
df.drop(columns=['NAMAPESERTA'], inplace=True)

# Remove the "KARYAWAN" column
df.drop(columns=['KARYAWAN'], inplace=True)

# Remove the "KLAIM" column
df.drop(columns=['KLAIM'], inplace=True)

# Remove the "TOLAK" column
df.drop(columns=['TOLAK'], inplace=True)

# Remove the "NAMAKODE" column
df.drop(columns=['NAMAKODE'], inplace=True)

# Remove the "KELASRAWAT" column
df.drop(columns=['KELASRAWAT'], inplace=True)

# Remove the "JUMLAHHARI" column
df.drop(columns=['JUMLAHHARI'], inplace=True)

# Remove the "KELASRI" column
df.drop(columns=['KELASRI'], inplace=True)

# ---- Start Feature Engineering ----
# Add a new column "STATUS" based on the "NOMORPESERTA" column
def get_status(row):
    last_char = row['NOMORPESERTA'][-1]
    if last_char == 'A':
        return '1' #Pasangan
    elif last_char == 'B': 
        return '2'#Anak
    else:
        return '3' #Pegawai


df['STATUS'] = df.apply(get_status, axis=1) 

# Remove the "PERIODE" column
df.drop(columns=['NOMORPESERTA'], inplace=True)

# Calculate the new column "HARGAPREMI", use the claim ratio (80%)
#df['HARGAPREMI'] =np.ceil(df['BAYAR'] * (100 / 80))

min_value = 50000  # Replace with your desired minimum value
max_value = 1000000000  # Replace with your desired maximum value

df['HARGAPREMI'] =np.random.randint(min_value, max_value, size=len(df))


unique_values = df['NAMAPERUSAHAAN'].unique()
value_to_code = {value: code for code, value in zip(range(1, 616), unique_values)}

value_mapping_JENISKELAMIN = {'L': 1, 'P': 2}
df['JENISKELAMIN'] = df['JENISKELAMIN'].map(value_mapping_JENISKELAMIN).fillna(0)

name_counts_perusahaan = df['NAMAPERUSAHAAN'].value_counts()
name_counts_penyakit = df['NAMAPENYAKIT'].value_counts()
name_counts_klaim = df['JENISKLAIM'].value_counts()

# Create a new column 'Total_Kemunculan' with name counts
df['FREKUENSIPERUSAHAAN'] = df['NAMAPERUSAHAAN'].map(name_counts_perusahaan)
# Create a new column 'Total_Kemunculan' with name counts
df['FREKUENSIPENYAKIT'] = df['NAMAPENYAKIT'].map(name_counts_penyakit)
# Create a new column 'Total_Kemunculan' with name counts
df['FREKUENSIKLAIM'] = df['JENISKLAIM'].map(name_counts_klaim)

# Data after cleaning
df.to_excel('clean.xlsx', index=False)
