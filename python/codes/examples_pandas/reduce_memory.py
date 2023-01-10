# !pip install --quiet fastai
import pandas as pd
from fastai.tabular.core import df_shrink

df = pd.read_csv(...) #shape (1000000, 100)
print(f"{df.memory_usage().sum()/(1024**2)} MB") 
#765.9395 MB


reduced_df = df_shrink(df, in2unit=True) #shape (1000000, 100)
print(f"{reduced_df.memory_usage().sum()/(1024**2)} MB") 
#381.4698 MB
