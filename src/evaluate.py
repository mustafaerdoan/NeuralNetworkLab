import torch
import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np

def calculate_perplexity(model, tokenizer, text, device):
    """Calculate perplexity of a text."""
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    return torch.exp(loss).item()

def evaluate_minimal_pairs(model_path, eval_dataset_path, output_path=None):
    """
    Evaluate a model on minimal pairs by comparing perplexities.
    A lower perplexity should indicate the model's preference for that sentence.
    """
    # Load model and tokenizer
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    model.to(device)
    model.eval()
    
    # Load evaluation data
    print(f"Loading evaluation data from {eval_dataset_path}")
    with open(eval_dataset_path, 'r') as f:
        eval_data = json.load(f)
    
    # Initialize results
    results = {
        "total_pairs": len(eval_data["minimal_pairs"]),
        "correct_predictions": 0,
        "phenomenon_accuracy": {},
        "phenomenon_counts": {},
        "pairs_with_predictions": []
    }
    
    # Evaluate each minimal pair
    print("Evaluating minimal pairs...")
    for pair in tqdm(eval_data["minimal_pairs"]):
        # Calculate perplexity for both sentences
        grammatical_ppl = calculate_perplexity(model, tokenizer, pair["grammatical"], device)
        ungrammatical_ppl = calculate_perplexity(model, tokenizer, pair["ungrammatical"], device)
        
        # The model predicts correctly if grammatical sentence has lower perplexity
        correct = grammatical_ppl < ungrammatical_ppl
        
        # Update phenomenon-specific accuracy
        phenomenon = pair["phenomenon"]
        if phenomenon not in results["phenomenon_accuracy"]:
            results["phenomenon_accuracy"][phenomenon] = 0
            results["phenomenon_counts"][phenomenon] = 0
        
        results["phenomenon_counts"][phenomenon] += 1
        if correct:
            results["correct_predictions"] += 1
            results["phenomenon_accuracy"][phenomenon] += 1
        
        # Store detailed results
        results["pairs_with_predictions"].append({
            "grammatical": pair["grammatical"],
            "ungrammatical": pair["ungrammatical"],
            "phenomenon": phenomenon,
            "grammatical_ppl": grammatical_ppl,
            "ungrammatical_ppl": ungrammatical_ppl,
            "correct": correct
        })
    
    # Calculate overall accuracy
    results["overall_accuracy"] = results["correct_predictions"] / results["total_pairs"]
    
    # Calculate phenomenon-specific accuracies
    for phenomenon in results["phenomenon_accuracy"]:
        results["phenomenon_accuracy"][phenomenon] /= results["phenomenon_counts"][phenomenon]
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
    print("\nAccuracy by phenomenon:")
    for phenomenon, accuracy in results["phenomenon_accuracy"].items():
        count = results["phenomenon_counts"][phenomenon]
        print(f"{phenomenon}: {accuracy:.2%} ({count} pairs)")
    
    # Save results if output path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {output_path}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a BabyLM model on minimal pairs.")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the model directory")
    parser.add_argument("--eval_dataset_path", type=str, required=True,
                      help="Path to the minimal pairs evaluation dataset")
    parser.add_argument("--output_path", type=str,
                      help="Optional path to save detailed results")
    args = parser.parse_args()
    
    evaluate_minimal_pairs(args.model_path, args.eval_dataset_path, args.output_path)