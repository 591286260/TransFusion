import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, TransformerConv

# Read the interaction relationships from the CSV file
df_interactions = pd.read_csv('../interaction.csv', header=None, names=['circRNA', 'miRNA'])

# Create graph structure
G = nx.Graph()

# Add nodes and edges
for _, row in df_interactions.iterrows():
    circRNA, miRNA = row['circRNA'], row['miRNA']
    G.add_node(circRNA, type='circRNA')
    G.add_node(miRNA, type='miRNA')
    G.add_edge(circRNA, miRNA)

# Convert the graph to PyTorch Geometric data format
node_to_index = {node: i for i, node in enumerate(G.nodes)}
edge_index = torch.tensor([[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in G.edges]).t().contiguous()

data = Data(edge_index=edge_index)

# Initialize node features as an identity matrix
num_nodes = G.number_of_nodes()
data.x = torch.eye(num_nodes)

# Print initialized node features to ensure correct initialization
print("Node feature matrix shape:", data.x.shape)
print("Sample of node features:", data.x[:5])

# Define Graph Transformer model
class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphTransformer, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)  # First Graph Convolution layer
        self.transformer_layer = TransformerConv(64, 64, heads=1)  # Transformer Layer
        self.conv2 = GCNConv(64, out_channels)  # Second Graph Convolution layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.transformer_layer(x, edge_index)  # Apply Transformer Layer
        x = self.conv2(x, edge_index)
        return x

# Set hyperparameters
in_channels = num_nodes
out_channels = 256  # Target feature dimension, adjustable as needed

# Initialize the model
model = GraphTransformer(in_channels, out_channels)

# Define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00014)
criterion = torch.nn.MSELoss()

# Generate target matrix, using the first 128 dimensions of the identity matrix as the target
identity = torch.eye(num_nodes, out_channels)

# Train the model
model.train()
for epoch in range(370):
    optimizer.zero_grad()
    out = model(data)
    target = identity[:, :out_channels]  # Use the first 128 dimensions of the identity matrix
    loss = criterion(out, target)  # Calculate loss
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Get node feature representations
model.eval()
node_embeddings = model(data).detach().numpy()

# Save node feature representations as a CSV file, without headers
node_embeddings_df = pd.DataFrame(node_embeddings, index=list(G.nodes))
node_embeddings_df.to_csv('node_embeddings.csv', header=False)
