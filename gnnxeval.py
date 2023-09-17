import streamlit as st

import torch
import torch_geometric
from torch_geometric.explain.metric import fidelity
from torch_geometric.explain.metric import characterization_score
from torch_geometric.explain.metric import unfaithfulness
from torch_geometric.nn import GATConv, GCNConv,GINConv,SAGEConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.contrib.explain import GraphMaskExplainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_metrics(selected_dataset, selected_model,selected_explainer):
    path = 'data\Planetoid'
    dataset = Planetoid(path, name= selected_dataset)
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(dataset.num_features, 16)
            self.conv2 = GCNConv(16, dataset.num_classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
    class GAT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(dataset.num_features, 16)
            self.conv2 = GATConv(16, dataset.num_classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
    class GIN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GINConv(dataset.num_features, 16)
            self.conv2 = GINConv(16, dataset.num_classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
    class GraphSAGE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(dataset.num_features, 16)
            self.conv2 = SAGEConv(16, dataset.num_classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
    model =GCN().to(device)

    if selected_model=="GCN":
        model = GCN().to(device)
    elif selected_model=="GAT":
        model = GAT().to(device)
    elif selected_model=="GIN":
        model = GIN().to(device)
    elif selected_model=="GraphSage":
        model = GraphSAGE().to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


    algorithm = GNNExplainer(epochs=200)

    if selected_explainer=="GraphMaskExplainer":
        algorithm =GraphMaskExplainer(2, epochs=5)
    elif selected_explainer=="GNNExplainer(epochs=200)":
        algorithm =GNNExplainer(epochs=200)

    topk = 40
    node_index=10

    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    threshold_config=dict(threshold_type  = 'topk', value=topk)
    )

    node_index = 10
    explanation = explainer(data.x, data.edge_index, index=node_index)


    explanation_metrics = explainer(data.x, data.edge_index)

    fid_pm = fidelity(explainer, explanation_metrics)
    st.write("Fidelity: "+ str(fid_pm))

    char_score = characterization_score(fid_pm[0], fid_pm[1])
    st.write("Characterization score: "+ str(char_score))

    unfaithfulness1=unfaithfulness(explainer,explanation,top_k=10)
    st.write("unfaithfulness: "+ str(unfaithfulness1))
    return explanation_metrics

# def get_feature_importance(explanation):
#     path_fi = 'feature_importance.png'
#     st.write(path_fi)
#     st.write(explanation)
#     explanation.visualize_feature_importance(path =path_fi, top_k=10)
#     st.image(path_fi ).show


# def get_subgraph(explanation):
#     path_sb = 'save_data\subgraph.pdf'
#     explanation.visualize_graph(path_sb)

def main():

    st.title("GNN xEval")
            
    
            
    options_dataset = ["CORA", "CiteSeer", "Pubmed"]
    options_explainer = ["GNNExplainer", "GraphMaskExplainer"]
    options_model = ["GCN","GAT", "GIN", "GraphSage"]
    selected_dataset = st.selectbox("Select a dataset", options_dataset)
    selected_explainer = st.selectbox("Select an Explainer", options_explainer)
    selected_model = st.selectbox("Select an GNN architecture", options_model)
    if st.button("Explain"):
        st.write(selected_dataset+", "+selected_explainer+", "+selected_model )
        explanation = get_metrics(selected_dataset,selected_explainer,selected_model)
        # if st.button("Feature_Importance"):
        #     get_feature_importance(explanation)
        # if st.button("Sub_Graph"):
        #     get_subgraph(explanation)    

if __name__ == "__main__":
    main()   