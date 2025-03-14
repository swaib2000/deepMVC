import torch
from models.ContrastiveMVC import ContrastiveMVC
from models.DSCN import DeepSubspaceClusteringNetwork
from utils.data_preprocessing import load_multiview_data
from clustering.SpectralClustering import spectral_clustering
from clustering.ContrastiveAlignment import contrastive_alignment
from utils.evaluation import evaluate_clustering

class MultiviewClusteringPipeline:
    def __init__(self, config):
        self.config = config

    def run(self):
        views = load_multiview_data(self.config['data_path'])
        mvc_model = ContrastiveMVC(self.config['input_dims'], self.config['embedding_dim'])
        embeddings = mvc_model(views)

        # Apply DSCN to refine embeddings
        dscn_model = DeepSubspaceClusteringNetwork(self.config['embedding_dim'], 256, self.config['latent_dim'])
        refined_embeddings = [dscn_model(embedding) for embedding in embeddings]

        # Cross-View Contrastive Alignment
        contrastive_loss = contrastive_alignment(refined_embeddings[0], refined_embeddings[1])

        # Final clustering
        final_embeddings = torch.cat(refined_embeddings, dim=1).detach().numpy()
        clusters = spectral_clustering(final_embeddings, n_clusters=self.config['n_clusters'])
        results = evaluate_clustering(true_labels=self.config['true_labels'], predicted_labels=clusters)
        print("Clustering Performance:", results)

