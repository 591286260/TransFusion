import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to extract feature vectors from input sequences using BERT
def extract_features(sequence):
    # Tokenize the sequence and convert it into input IDs
    inputs = tokenizer.encode_plus(sequence, add_special_tokens=True, return_tensors="pt")
    # Extract feature vectors using the BERT model
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
    # Get the [CLS] token's output as the sequence feature vector
    features = outputs.last_hidden_state[:, 0, :16]  # Modify this slice as needed
    return features.detach().numpy().tolist()[0]

# Load circRNA data
circrna_df = pd.read_csv("circRNA.csv", header=None, names=["id", "sequence"])
circrna_df["sequence"] = circrna_df["sequence"].apply(lambda x: x[:90])
# Extract feature vectors for circRNA sequences
circrna_df["vector"] = circrna_df["sequence"].apply(lambda x: extract_features(x))

# Split feature vectors into separate columns
circrna_df = pd.concat([circrna_df["id"], circrna_df["vector"].apply(pd.Series)], axis=1)

# Save the circRNA feature vectors to a CSV file
circrna_df.to_csv("circRNA.csv", index=False, header=False)
