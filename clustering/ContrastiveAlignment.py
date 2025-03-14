def contrastive_alignment(embeddings_view1, embeddings_view2, tau=0.5):
    cos_sim = torch.nn.functional.cosine_similarity(embeddings_view1, embeddings_view2)
    loss = -torch.log(torch.exp(cos_sim / tau).sum())
    return loss