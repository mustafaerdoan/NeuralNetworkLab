import json
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(model1_results_path, model2_results_path, output_path):
    # Load results
    with open(model1_results_path, 'r') as f:
        model1_results = json.load(f)
    with open(model2_results_path, 'r') as f:
        model2_results = json.load(f)
    
    # Prepare data for plotting
    phenomena = list(model1_results["phenomenon_accuracy"].keys())
    model1_acc = [model1_results["phenomenon_accuracy"][p] * 100 for p in phenomena]
    model2_acc = [model2_results["phenomenon_accuracy"][p] * 100 for p in phenomena]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Set width of bars and positions of the bars
    width = 0.35
    x = np.arange(len(phenomena))
    
    # Create bars
    plt.bar(x - width/2, model1_acc, width, label='Model 1 (Small)', alpha=0.8)
    plt.bar(x + width/2, model2_acc, width, label='Model 2 (Large)', alpha=0.8)
    
    # Customize the plot
    plt.ylabel('Accuracy (%)')
    plt.title('Model Performance Comparison by Linguistic Phenomenon')
    plt.xticks(x, phenomena, rotation=45, ha='right')
    plt.legend()
    
    # Add overall accuracy in the title
    plt.title(f'Model Performance Comparison\nModel 1 Overall: {model1_results["overall_accuracy"]*100:.1f}% | Model 2 Overall: {model2_results["overall_accuracy"]*100:.1f}%')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    plot_comparison(
        "results/model1_evaluation.json",
        "results/model2_evaluation.json",
        "results/model_comparison.png"
    )
