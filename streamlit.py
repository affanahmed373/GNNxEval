import torch
import torch_geometric
import streamlit
import requests
from fastapi import FastAPI
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pickle
# import torchmetrics
from torch_geometric.explain.metric import groundtruth_metrics
from torch_geometric.explain.metric import fidelity
from torch_geometric.explain.metric import characterization_score
from torch_geometric.explain.metric import fidelity_curve_auc
from torch_geometric.explain.metric import unfaithfulness

epsilon = 0.999

import streamlit as st

def main():
    st.title("GNN xEval")
    targets =[]
    preds =[]
    faith = []
    fid_pos = []
    fid_neg =[]
    auc= 0
    select ='a'

    options_dataset = ["CORA", "CiteSeer", "BAshapes", "KarateClub"]
    options_explainer = ["GNNExplainer", "PGExplainer", "Integrated Gradients", "Saliency"]
    options_architecture = ["GCN", "GIN", "GraphSage","GAT"]
    options_task = ["Node_Classification", "Link Prediction", "Graph Classification"]
    
    selected_dataset = st.selectbox("Select a dataset", options_dataset)
    selected_explainer = st.selectbox("Select an Explainer", options_explainer)
    selected_architecture = st.selectbox("Select an GNN architecture", options_architecture)
    selected_task = st.selectbox("Select the GNN task", options_task)
    
    
    
    if st.button("Load"):
        st.write(selected_dataset+', '+selected_explainer+', '+selected_architecture)

    
        select ='none'
        # ["GNNExplainer", "PGExplainer", "Integrated Gradients", "Saliency","Occlusion","AttentionExplainer", "PGMExplainer"]
        if selected_dataset== "CORA":
            select = 'C'
        if selected_dataset== "CiteSeer":
            select = 'M'
        if selected_dataset== "BAshapes":
            select ='B'
        if selected_dataset== "KarateClub":
            select ='K'

      

        if selected_explainer =="GNNExplainer":
            select+='G'
        if selected_explainer =="PGExplainer":
            select+="P"
        if selected_explainer =="Integrated Gradients":
            select+= "I"
        if selected_explainer =="Saliency":
            select+="S"         

        if selected_architecture =="GCN":
            select +='C'
        if selected_architecture =="GAT":
            select +='A'
        if selected_architecture =="GIN":
            select +='I'
        if selected_architecture =="GraphSage":
            select +='S'  

        # loaded_data = torch.load(select+'.pth')
        # loaded_data = torch.load('models\CGC.pth')
        loaded_data = torch.load('models/111.pth')

# Access the individual tensors and integer
        unfaithfulness1 = loaded_data['unfaithfulness']
        explanation = loaded_data['explanation']
        Charac= loaded_data['Charac']
        fid_pm = loaded_data['fid_pm']
# Use the loaded variables
        explanation.visualize_feature_importance( top_k=10)



    st.write("Characterisation score: ")
    st.write(Charac)
    st.write("Unfaithfullness: ")
    st.write(unfaithfulness1)
    st.write("Fidelity: ")
    st.write(fid_pm)






    # fpr, tpr, thresholds = roc_curve(torch.cat(targets), torch.cat(preds))
    # plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    # fig = plt.figure(figsize=(8,8))
    # plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc))
    # plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random predictions
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('AUC-ROC Curve')
    # plt.legend()
    # st.pyplot(fig)

if __name__ == "__main__":
    main()