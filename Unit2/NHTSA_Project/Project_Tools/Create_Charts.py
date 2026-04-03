import pandas as pd
import matplotlib.pyplot as plt
import os

def create_bar_chart(csv_path, columns, output_dir):
    """
    Reads a CSV file, creates a bar chart for the counts of unique entries
    in each specified column, and saves them to output_dir/charts/.
    """
    charts_dir = os.path.join(output_dir, "charts")
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in {csv_path}. Skipping.")
            continue
            
        counts = df[col].value_counts()
        
        plt.figure(figsize=(10, 6))
        counts.plot(kind='bar')
        plt.title(f"Counts of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        
        output_file = os.path.join(charts_dir, f"{col}_bar_chart.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Saved bar chart for '{col}' to {output_file}")
