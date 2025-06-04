import os
import torch
from gensim import models
from . import global_vars
from torch_geometric.nn import SAGEConv, to_hetero


# Get the directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

graph_path = os.path.join(BASE_DIR, 'recFiles', 'web_graph.pt')
data = torch.load(graph_path)
global_vars.graph_path=graph_path


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class CosineSimilarityDecoder(torch.nn.Module):
    def forward(self, x_dict, edge_label_index):
        x_src = x_dict['user'][edge_label_index[0]]
        x_dst = x_dict['job'][edge_label_index[1]]
        return torch.cosine_similarity(x_src, x_dst, dim=1)

class Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels):
        super().__init__()
        self.encoder = GCN(input_dim, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = CosineSimilarityDecoder()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = self.encoder(x_dict, edge_index_dict)
        cosine_similarity = self.decoder(x_dict, edge_label_index)
        return cosine_similarity
    

# Path to the model file
model_path = os.path.join(BASE_DIR, 'recFiles', 'model_dict.pt')
lda_model_path = os.path.join(BASE_DIR, 'recFiles', 'lda_model_25')
skill_path = os.path.join(BASE_DIR, 'recFiles', 'skill_ruler.jsonl')
degree_path = os.path.join(BASE_DIR, 'recFiles', 'degree_ruler.jsonl')
major_path = os.path.join(BASE_DIR, 'recFiles', 'majors_ruler.jsonl')

# Load the models
model=Model(input_dim=768, hidden_channels=32)
model.load_state_dict(torch.load(model_path))
global_vars.model = model
global_vars.LDA_model = models.LdaModel.load(lda_model_path)
global_vars.skill_path = skill_path
global_vars.degree_path = degree_path
global_vars.major_path = major_path
