from datasets import load_dataset
import random
import argparse
from pathlib import Path

def prepare_data(output_file, size_mb=2, split="train"):
    """
    Prepare training data from Wikitext-2 dataset.
    Args:
        output_file: Path to save the prepared data
        size_mb: Size of the output file in megabytes
        split: Dataset split to use (train/validation/test)
    """
    # Load Wikitext-2 dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")[split]
    
    # Calculate bytes per character (approximate)
    bytes_per_char = 1  # assuming ASCII/UTF-8
    target_chars = int(size_mb * 1024 * 1024 // bytes_per_char)
    
    # Prepare text data
    all_text = "\n".join(dataset["text"])
    
    # If we need less data than available, randomly select a subset
    if len(all_text) > target_chars:
        # Find a random starting point
        max_start = int(len(all_text) - target_chars)
        start_idx = random.randint(0, max_start)
        # Extract the substring
        selected_text = all_text[start_idx:start_idx + target_chars]
        # Adjust to not cut in the middle of a line
        selected_text = selected_text[selected_text.find("\n")+1:selected_text.rfind("\n")]
    else:
        selected_text = all_text
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(selected_text)
    
    print(f"Saved {len(selected_text) / (1024*1024):.2f}MB of text to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data from Wikitext-2")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the prepared data")
    parser.add_argument("--size_mb", type=float, default=2, help="Size of the output file in megabytes")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"], 
                      help="Dataset split to use")
    args = parser.parse_args()
    
    prepare_data(args.output_file, args.size_mb, args.split)
