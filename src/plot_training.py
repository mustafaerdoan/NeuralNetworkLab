import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def plot_training_logs(model_path):
    # Read the training logs
    logs_path = Path(model_path) / 'training_logs.csv'
    df = pd.read_csv(logs_path)
    
    # Set up the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Plot training loss
    train_data = df[df['loss'].notna()]
    plt.plot(train_data['epoch'], train_data['loss'], label='Training Loss', alpha=0.6)
    
    # Plot evaluation loss
    eval_data = df[df['eval_loss'].notna()]
    plt.plot(eval_data['epoch'], eval_data['eval_loss'], label='Validation Loss', alpha=0.6)
    
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = Path(model_path) / 'training_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training logs")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the model directory containing training_logs.csv")
    args = parser.parse_args()
    plot_training_logs(args.model_path)
