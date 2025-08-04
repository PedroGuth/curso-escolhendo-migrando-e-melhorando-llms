import pandas as pd
import random
from datasets import load_dataset
import os


# Set random seed for reproducibility
random.seed(42)

# Load the XSum dataset
print("Loading XSum dataset...")
ds = load_dataset("EdinburghNLP/xsum")

# XSum has train, validation, and test splits
# We'll sample from the validation set to avoid using training data
validation_set = ds['validation']

# Convert to a pandas DataFrame for easier handling
print("Converting to DataFrame...")
df = pd.DataFrame({
    'document': validation_set['document'],
    'referenceResponse': validation_set['summary']
})

# Create samples of different sizes
print("Creating samples...")
sample_sizes = [10, 50, 100]

# Create output directory if it doesn't exist
os.makedirs('xsum_samples', exist_ok=True)

# Create and save each sample
for size in sample_sizes:
    print(f"Sampling {size} examples...")
    
    # Take a random sample of the specified size
    sample_df = df.sample(size, random_state=42)
    
    # Save to CSV
    output_file = f'data/xsum_sample_{size}.csv'
    sample_df.to_csv(output_file, index=False)
    print(f"Saved {size} samples to {output_file}")
    
    # Print some stats
    avg_doc_length = sample_df['document'].str.len().mean()
    avg_summary_length = sample_df['referenceResponse'].str.len().mean()
    print(f"  - Average document length: {avg_doc_length:.1f} characters")
    print(f"  - Average summary length: {avg_summary_length:.1f} characters")

print("Sampling complete! Files saved in the xsum_samples directory.")

