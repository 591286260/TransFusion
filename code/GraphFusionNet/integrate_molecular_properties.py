import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, TransformerConv

# Read interaction relationships
df_interactions = pd.read_csv('interaction.csv', header=None, names=['circRNA', 'miRNA'])
G = nx.Graph()

# Add nodes and edges
for _, row in df_interactions.iterrows():
    circRNA, miRNA = row['circRNA'], row['miRNA']
    G.add_node(circRNA, type='circRNA')
    G.add_node(miRNA, type='miRNA')
    G.add_edge(circRNA, miRNA)

# Load node features
df_features = pd.read_csv('../3_sequence_features/Se_vector256.csv', header=None)
df_features.columns = ['id'] + [f'feature_{i}' for i in range(1, df_features.shape[1])]

node_features = df_features.set_index('id').T.to_dict('list')

# Convert graph to PyTorch Geometric format
node_to_index = {node: i for i, node in enumerate(G.nodes)}
edge_index = torch.tensor([[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in G.edges]).t().contiguous()

data = Data(edge_index=edge_index)

# Initialize node features
num_nodes = G.number_of_nodes()
feature_dim = len(next(iter(node_features.values())))
data.x = torch.zeros((num_nodes, feature_dim))

# Assign features to nodes
for node, features in node_features.items():
    if node in node_to_index:
        data.x[node_to_index[node]] = torch.tensor(features, dtype=torch.float)

# Define Graph Transformer model
class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphTransformer, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.transformer_layer = TransformerConv(64, 64, heads=1)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.transformer_layer(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

# Set hyperparameters
in_channels = feature_dim
out_channels = feature_dim

# Initialize model
model = GraphTransformer(in_channels, out_channels)

# Define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00014)
criterion = torch.nn.MSELoss()

# Train model
model.train()
for epoch in range(370):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.x)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Get node embeddings
model.eval()
node_embeddings = model(data).detach().numpy()

# Save node embeddings to CSV
node_embeddings_df = pd.DataFrame(node_embeddings, index=list(G.nodes))
node_embeddings_df.to_csv('维度对比/node_embeddings256.csv', header=False)
