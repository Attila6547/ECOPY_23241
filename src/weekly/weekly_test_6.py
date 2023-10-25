import pandas as pd

df_sp500 = pd.read_parquet('C:/Users/Atti60/Documents/GitHub/ECOPY_23241/data/sp500.parquet', engine='fastparquet')
df_factors = pd.read_parquet('C:/Users/Atti60/Documents/GitHub/ECOPY_23241/data/ff_factors.parquet', engine='fastparquet')

merged_df = pd.merge(df_sp500, df_factors, on='Date', how='left')
merged_df['Excess Return'] = merged_df['Monthly Returns'] - merged_df['RF']

merged_df['ex_ret_1'] = merged_df.groupby('Symbol')['Excess Return'].shift(-1)

merged_df.dropna(subset=['ex_ret_1', 'HML'], inplace=True)





