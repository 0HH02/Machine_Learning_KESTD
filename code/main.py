import torch
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# Datos: Grafos con nodos protagonistas
graphs = [
    Data(x=torch.rand(5, 3), edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long), 
         mask=torch.tensor([0, 1, 0, 1, 0], dtype=torch.bool), y=torch.tensor([1.5, 2.5])),
    Data(x=torch.rand(6, 3), edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long), 
         mask=torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool), y=torch.tensor([3.0, 4.5]))
]

loader = DataLoader(graphs, batch_size=2, shuffle=True)

# Definir el modelo utilizando capas predefinidas
class GNNRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNConv(3, 16)
        self.gcn2 = GCNConv(16, 16)
        self.regressor = torch.nn.Linear(16, 1)

    def forward(self, x, edge_index, mask):
        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        return self.regressor(x[mask]).squeeze(-1)

# Entrenar el modelo
model = GNNRegressor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

for epoch in range(10):
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.mask)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
