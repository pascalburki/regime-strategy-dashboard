import yfinance as yf

df_full = yf.download('NG=F', start='2000-01-01', end='2023-12-31')
print(df_full.index.min())
print(len(df_full))