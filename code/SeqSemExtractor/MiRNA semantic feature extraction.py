import torch
from transformers import ElectraTokenizer, ElectraModel
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from tqdm import tqdm

# Check for available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load miRNA data
miRNA_df = pd.read_csv('miRNA.csv', header=None)
miRNA_ids = miRNA_df.iloc[:, 0].tolist()  # Extract ID column
miRNA_sequences = miRNA_df.iloc[:, 1].tolist()  # Extract sequence column

# Load the ELECTRA model and tokenizer
model_name = 'google/electra-base-discriminator'
tokenizer = ElectraTokenizer.from_pretrained(model_name)
model = ElectraModel.from_pretrained(model_name)
model.to(device)

# Function to extract features with a maximum sequence length of 96
def extract_features(sequence):
    encoded_input = tokenizer(sequence, padding='max_length', truncation=True, max_length=96, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model(**encoded_input)
        features = output.last_hidden_state.mean(dim=1).squeeze()
    return features.tolist()

# Create a thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

# Process miRNA sequences in batches
miRNA_features = []
print("Processing miRNA sequences:")
with tqdm(total=len(miRNA_sequences)) as pbar:
    futures = [executor.submit(extract_features, sequence) for sequence in miRNA_sequences]
    for future in futures:
        miRNA_features.append(future.result())
        pbar.update(1)

# Create a DataFrame with feature data and IDs
df = pd.DataFrame(miRNA_features, columns=[f"Feature_{i+1}" for i in range(16)])  # Set feature column names
df.insert(0, "ID", miRNA_ids)  # Insert ID column at the beginning

# Save the features to a CSV file
df.to_csv('miRNA.csv', index=False, header=False)
