from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def get_visualization(model, train_dir: str):
    tc = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    full_dataset = datasets.ImageFolder(root=train_dir, transform=tc)
    model.to('cpu')
    layer = model._modules.get('avgpool')

    outputs = []

    def copy_embeddings(m, i, o):
        o = o[:, :, 0, 0].detach().numpy().tolist()
        outputs.append(o)

    _ = layer.register_forward_hook(copy_embeddings)

    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True, num_workers=2)

    cls = []

    model.eval()
    for X, y in full_loader:
        label = y.tolist()
        cls.append(label)
        _ = model(X)

    cls = [item for sublist in cls for item in sublist]
    list_embeddings = [item for sublist in outputs for item in sublist]
    features = np.array(list_embeddings)
    while (1):
        print("Select: 1:2D-PCA, 2:3D-PCA, 3:2D-TSNE,\n,4:3D-TSNE, 5:2D-UMAP, 6:3D-UMAP, exit:another button")
        i = str(input())
        if i == '1':
            pca = PCA(n_components=2)
            pca.fit(features)
            pca_features = pca.transform(features)
            plot_2d(pca_features[:, 0], pca_features[:, 1], cls)
        elif i == '2':
            pca = PCA(n_components=3)
            pca.fit(features)
            pca_features = pca.transform(features)
            plot_3d(pca_features[:, 0], pca_features[:, 1], pca_features[:, 2], cls)
        elif i == '3':
            tsne = TSNE(random_state=42, n_components=2, verbose=0, perplexity=25, n_iter=500).fit_transform(features)
            plot_2d(tsne[:, 0], tsne[:, 1], cls)
        elif i == '4':
            tsne = TSNE(random_state=42, n_components=3, verbose=0, perplexity=25, n_iter=500).fit_transform(features)
            plot_3d(tsne[:, 0], tsne[:, 1], tsne[:, 2], cls)
        elif i == '5':
            reducer = umap.UMAP(random_state=42, n_components=2)
            embedding = reducer.fit_transform(features)
            plot_2d(reducer.embedding_[:, 0], reducer.embedding_[:, 1], cls)
        elif i == '6':
            reducer = umap.UMAP(random_state=42, n_components=3)
            embedding = reducer.fit_transform(features)
            plot_3d(reducer.embedding_[:, 0], reducer.embedding_[:, 1], reducer.embedding_[:, 2], cls)
        else:
            return 0


def plot_2d(component1, component2, cls):
    fig = go.Figure(data=go.Scatter(
        x=component1,
        y=component2,
        mode='markers',
        marker=dict(
            size=20,
            color=cls,
            colorscale='Rainbow',
            showscale=True,
            line_width=1
        )
    ))
    fig.update_layout(margin=dict(l=100, r=100, b=100, t=100), width=2000, height=1200)
    fig.layout.template = 'plotly_dark'
    fig.show()


def plot_3d(component1, component2, component3, cls):
    fig = go.Figure(data=[go.Scatter3d(
        x=component1,
        y=component2,
        z=component3,
        mode='markers',
        marker=dict(
            size=10,
            color=cls,
            colorscale='Rainbow',
            opacity=1,
            line_width=1
        )
    )])

    fig.update_layout(margin=dict(l=50, r=50, b=50, t=50), width=1800, height=1000)
    fig.layout.template = 'plotly_dark'
    fig.show()
