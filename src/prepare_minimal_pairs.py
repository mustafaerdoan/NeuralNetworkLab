from datasets import load_dataset
import json
import random
import argparse
from pathlib import Path

def prepare_minimal_pairs(output_file, num_pairs=1000, phenomena=None):
    """
    Prepare minimal pairs from BLiMP dataset.
    Args:
        output_file: Path to save the prepared minimal pairs
        num_pairs: Number of minimal pairs to select
        phenomena: List of linguistic phenomena to include (if None, select from all)
    """
    # Load BLiMP dataset
    # Select a subset of phenomena to keep the task focused
    default_phenomena = [
        'determiner_noun_agreement_1',
        'determiner_noun_agreement_2',
        'regular_plural_subject_verb_agreement_1',
        'regular_plural_subject_verb_agreement_2',
        'anaphor_number_agreement',
        'anaphor_gender_agreement'
    ]
    
    phenomena = phenomena if phenomena else default_phenomena
    
    # Load and combine datasets for each phenomenon
    all_data = []
    for phenomenon in phenomena:
        try:
            data = load_dataset("blimp", phenomenon)["train"]
            all_data.extend([{**item, "phenomenon": phenomenon} for item in data])
        except ValueError as e:
            print(f"Warning: Could not load phenomenon '{phenomenon}': {e}")
            continue
    
    if not all_data:
        raise ValueError("No data could be loaded from any of the specified phenomena")
    
    # Ensure we have enough pairs
    total_pairs = len(all_data)
    if total_pairs < num_pairs:
        print(f"Warning: Only {total_pairs} pairs available, using all of them")
        num_pairs = total_pairs
    
    # Randomly select pairs
    selected_indices = random.sample(range(total_pairs), num_pairs)
    selected_pairs = [all_data[i] for i in selected_indices]
    
    # Format pairs for output
    formatted_pairs = []
    for pair in selected_pairs:
        formatted_pairs.append({
            "grammatical": pair["sentence_good"],
            "ungrammatical": pair["sentence_bad"],
            "phenomenon": pair["phenomenon"]
        })
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "minimal_pairs": formatted_pairs,
            "metadata": {
                "num_pairs": len(formatted_pairs),
                "phenomena": list(set(p["phenomenon"] for p in formatted_pairs))
            }
        }, f, indent=2)
    
    print(f"Saved {len(formatted_pairs)} minimal pairs to {output_file}")
    print("Phenomena included:")
    for phenomenon in sorted(set(p["phenomenon"] for p in formatted_pairs)):
        count = sum(1 for p in formatted_pairs if p["phenomenon"] == phenomenon)
        print(f"  - {phenomenon}: {count} pairs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare minimal pairs from BLiMP dataset")
    parser.add_argument("--output_file", type=str, required=True,
                      help="Path to save the prepared minimal pairs")
    parser.add_argument("--num_pairs", type=int, default=1000,
                      help="Number of minimal pairs to select")
    parser.add_argument("--phenomena", type=str, nargs="+",
                      help="List of linguistic phenomena to include")
    args = parser.parse_args()
    
    prepare_minimal_pairs(args.output_file, args.num_pairs, args.phenomena)
