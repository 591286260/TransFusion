import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load feature vectors from CSV, without headers
data = pd.read_csv('node_embeddings3.csv', header=None)

# Extract IDs and feature vectors
names = data.iloc[:, 0].values  # First column is ID
X = data.iloc[:, 1:].values      # Remaining columns are features

# Check if feature vector data is empty
if X.shape[1] == 0:
    raise ValueError("No feature vectors found in the input file")

# Create labels based on the ID
labels = ['circRNA' if name.startswith('hsa_circ') else 'miRNA' for name in names]

# Perform t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Create a DataFrame to store the reduced data and labels
df_tsne = pd.DataFrame({
    'Name': names,
    'Dimension 1': X_tsne[:, 0],
    'Dimension 2': X_tsne[:, 1],
    'Label': labels
})

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Plotting
plt.figure(figsize=(10, 8))

# Scatter points for circRNA and miRNA
circRNA_data = df_tsne[df_tsne['Label'] == 'circRNA']
miRNA_data = df_tsne[df_tsne['Label'] == 'miRNA']
plt.scatter(circRNA_data['Dimension 1'], circRNA_data['Dimension 2'],
            c=(226/255, 167/255, 185/255), label='circRNA', alpha=0.7)
plt.scatter(miRNA_data['Dimension 1'], miRNA_data['Dimension 2'],
            c=(172/255, 203/255, 208/255), label='miRNA', alpha=0.7)

# Set axis tick parameters
plt.tick_params(axis='x', labelsize=17)
plt.tick_params(axis='y', labelsize=17)


plt.legend(markerscale=1.8, fontsize=17)  # Legend settings
plt.show()
