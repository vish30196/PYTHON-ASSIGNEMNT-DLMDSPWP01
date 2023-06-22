import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
ideal_df = pd.read_csv('ideal.csv')

train_df

test_df

ideal_df

plt.figure(figsize=(15, 6))
sns.lineplot(x=train_df['x'], y=train_df['y1'], data=train_df, palette='hls')
sns.scatterplot(x=train_df['x'], y=train_df['y1'], data=train_df, palette='hls', color="red")
plt.show()

plt.figure(figsize=(15, 6))
sns.lineplot(x=test_df['x'], y=test_df['y'], data=test_df, palette='hls')
sns.scatterplot(x=test_df['x'], y=test_df['y'], data=test_df, palette='hls', color="red")
plt.show()
