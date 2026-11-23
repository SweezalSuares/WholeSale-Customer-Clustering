"""Core clustering logic (ported from the lab notebook). No Streamlit imports here."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

METRICS = ["euclidean", "cityblock", "cosine"]
LINKAGES = ["single", "complete", "average", "ward"]
CATEGORICAL = ["Channel", "Region"]


def prepare(df: pd.DataFrame):
    """Split meta columns, keep spend features, standardise."""
    meta = df[[c for c in CATEGORICAL if c in df.columns]].reset_index(drop=True)
    X = df.drop(columns=[c for c in CATEGORICAL if c in df.columns]).reset_index(drop=True)
    X = X.select_dtypes(include=[np.number])
    X_scaled = StandardScaler().fit_transform(X)
    return meta, X, X_scaled


def distance_matrices(X_scaled):
    vec = {m: pdist(X_scaled, metric=m) for m in METRICS}
    mat = {m: squareform(v) for m, v in vec.items()}
    return vec, mat


def hierarchical_all(X_scaled, K, dist_vec):
    """All metric x linkage combos (Ward only with Euclidean)."""
    rows, labels, Zs = [], {}, {}
    for metric in METRICS:
        for link in LINKAGES:
            if link == "ward" and metric != "euclidean":
                continue
            Z = linkage(X_scaled, method=link, metric=metric)
            lab = fcluster(Z, K, criterion="maxclust") - 1
            Zs[(metric, link)] = Z
            labels[(metric, link)] = lab
            coph, _ = cophenet(Z, dist_vec[metric])
            sizes = np.bincount(lab, minlength=K)
            n_unique = len(np.unique(lab))
            if n_unique > 1:
                sil = silhouette_score(X_scaled, lab,
                                       metric="manhattan" if metric == "cityblock" else metric)
                db = davies_bouldin_score(X_scaled, lab)
                ch = calinski_harabasz_score(X_scaled, lab)
            else:
                sil = db = ch = np.nan
            rows.append(dict(Metric=metric, Linkage=link, Silhouette=sil,
                             DaviesBouldin=db, CalinskiHarabasz=ch,
                             Cophenetic=coph, ClusterSizes=sizes.tolist()))
    return pd.DataFrame(rows).round(3), labels, Zs


def divisive(X_scaled, K, seed=42):
    model = BisectingKMeans(n_clusters=K, bisecting_strategy="biggest_inertia",
                            random_state=seed, n_init=10)
    return model.fit_predict(X_scaled)


def scores(X_scaled, lab):
    return dict(Silhouette=silhouette_score(X_scaled, lab),
                DaviesBouldin=davies_bouldin_score(X_scaled, lab),
                CalinskiHarabasz=calinski_harabasz_score(X_scaled, lab))


def explore_k(X_scaled, k_min=2, k_max=8, seed=42):
    out = []
    for k in range(k_min, k_max + 1):
        for name, model in [("Ward", AgglomerativeClustering(n_clusters=k, linkage="ward")),
                            ("Average", AgglomerativeClustering(n_clusters=k, linkage="average")),
                            ("Divisive", BisectingKMeans(n_clusters=k, random_state=seed, n_init=10))]:
            lab = model.fit_predict(X_scaled)
            out.append(dict(k=k, Method=name,
                            Silhouette=silhouette_score(X_scaled, lab),
                            CH=calinski_harabasz_score(X_scaled, lab),
                            Sizes=sorted(np.bincount(lab).tolist(), reverse=True)))
    return pd.DataFrame(out)


def ari_matrix(labels, div_lab):
    keys = list(labels.keys()) + [("bisecting", "divisive")]
    labs = {**labels, ("bisecting", "divisive"): div_lab}
    names = [f"{m[:4]}-{l}" for m, l in keys]
    return pd.DataFrame([[adjusted_rand_score(labs[a], labs[b]) for b in keys] for a in keys],
                        index=names, columns=names).round(2)


def pca_2d(X_scaled, features):
    pca = PCA(n_components=2)
    P = pca.fit_transform(X_scaled)
    loadings = pd.DataFrame(pca.components_.T, index=features, columns=["PC1", "PC2"]).round(3)
    return P, pca.explained_variance_ratio_, loadings


def profile(X, meta, lab):
    prof = X.copy()
    prof["Cluster"] = lab
    out = prof.groupby("Cluster").mean().round(0)
    out["Size"] = prof.groupby("Cluster").size()
    if "Channel" in meta.columns:
        out["%Horeca (Ch1)"] = meta.groupby(lab)["Channel"].apply(lambda s: (s == 1).mean() * 100).round(1)
    return out
