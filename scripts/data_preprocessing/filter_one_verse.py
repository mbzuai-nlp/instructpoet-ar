import pandas as pd
import os

train_path = '/path/to/data/train_data.csv'

# Read the CSV
df = pd.read_csv(train_path)

# Filter out rows with exactly 1 verse (keep rows with != 1 verse)
df_filtered = df[df['poem_verses'] != 1]

# Save in same directory
out_path = os.path.join(os.path.dirname(train_path), 'train_set_one_verse_filtered.csv')
df_filtered.to_csv(out_path, index=False)

print(f"Original rows: {len(df)}")
print(f"Filtered rows: {len(df_filtered)}")
print(f"Saved to: {out_path}")
