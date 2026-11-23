import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster

import clustering as cl

st.set_page_config(page_title="Wholesale Customer Clustering", page_icon="📊", layout="wide")

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "Wholesale customers data.csv")

# ---------------------------------------------------------------- data loading
st.sidebar.title("⚙️ Settings")
up = st.sidebar.file_uploader("Upload 'Wholesale customers data.csv'", type=["csv"])

if up is not None:
    df = pd.read_csv(up)
elif os.path.exists(DEFAULT_CSV):
    df = pd.read_csv(DEFAULT_CSV)
    st.sidebar.caption("Using bundled dataset.")
else:
    st.title("📊 Wholesale Customer Clustering")
    st.info("Upload **Wholesale customers data.csv** in the sidebar to begin "
            "(or place it next to `app.py`).")
    st.stop()

missing = [c for c in cl.CATEGORICAL if c not in df.columns]
if missing:
    st.warning(f"Columns {missing} not found; running without them.")

K = st.sidebar.slider("Number of clusters (k)", 2, 8, 3)
metric = st.sidebar.selectbox("Distance metric", cl.METRICS)
link_opts = cl.LINKAGES if metric == "euclidean" else [l for l in cl.LINKAGES if l != "ward"]
linkage_choice = st.sidebar.selectbox("Linkage", link_opts, index=link_opts.index("ward") if "ward" in link_opts else 0)
seed = st.sidebar.number_input("Random seed (divisive)", 0, 9999, 42)
st.sidebar.caption("Ward linkage is only defined for Euclidean distance.")

# ---------------------------------------------------------------- computation (cached)
@st.cache_data(show_spinner="Preparing data…")
def _prep(d):
    meta, X, Xs = cl.prepare(d)
    vec, mat = cl.distance_matrices(Xs)
    return meta, X, Xs, vec, mat

@st.cache_data(show_spinner="Running hierarchical clustering…")
def _hier(Xs, K, vec):
    return cl.hierarchical_all(Xs, K, vec)

@st.cache_data(show_spinner="Running divisive clustering…")
def _div(Xs, K, seed):
    return cl.divisive(Xs, K, seed)

@st.cache_data(show_spinner="Exploring k = 2…8 (this takes a few seconds)…")
def _explore(Xs, seed):
    return cl.explore_k(Xs, 2, 8, seed)

meta, X, Xs, vec, mat = _prep(df)
features = list(X.columns)
res, labels, Zs = _hier(Xs, K, vec)
div_lab = _div(Xs, K, seed)
ward_lab = labels[("euclidean", "ward")]
chosen_lab = labels[(metric, linkage_choice)]

# ---------------------------------------------------------------- header
st.title("📊 Wholesale Customer Clustering")
st.caption("Agglomerative (hierarchical) vs divisive (Bisecting K-Means) clustering of annual spend "
           "across product categories.")

tabs = st.tabs(["Data", "Preprocessing", "Distances", "Hierarchical results", "Dendrograms",
                "Divisive vs Ward", "Explore k", "PCA view", "Agreement (ARI)", "Cluster profiles"])

# ---- 1. Data
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0]); c2.metric("Columns", df.shape[1])
    c3.metric("Duplicate rows", int(df.duplicated().sum()))
    st.subheader("Head"); st.dataframe(df.head(), use_container_width=True)
    st.subheader("Tail"); st.dataframe(df.tail(), use_container_width=True)
    st.subheader("Summary statistics"); st.dataframe(df.describe(), use_container_width=True)
    a, b = st.columns(2)
    a.subheader("Data types"); a.dataframe(df.dtypes.astype(str).rename("dtype"))
    b.subheader("Missing values"); b.dataframe(df.isnull().sum().rename("missing"))

# ---- 2. Preprocessing
with tabs[1]:
    st.write("**Dropped (categorical codes):**", ", ".join(meta.columns) or "none")
    st.write("**Features used:**", ", ".join(features))
    st.subheader("After StandardScaler")
    st.dataframe(pd.DataFrame({"mean": Xs.mean(axis=0).round(3), "std": Xs.std(axis=0).round(3)},
                              index=features))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    X.boxplot(ax=ax[0]); ax[0].set_title("Raw spend"); ax[0].tick_params(axis="x", rotation=30)
    pd.DataFrame(Xs, columns=features).boxplot(ax=ax[1]); ax[1].set_title("Scaled")
    ax[1].tick_params(axis="x", rotation=30)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ---- 3. Distances
with tabs[2]:
    for m in cl.METRICS:
        D, v = mat[m], vec[m]
        st.subheader(f"{m.capitalize()} distance")
        st.caption(f"shape={D.shape} · mean={v.mean():.3f} · min={v.min():.3f} · max={v.max():.3f}")
        st.dataframe(pd.DataFrame(D[:5, :5]).round(3), use_container_width=False)

# ---- 4. Hierarchical results
with tabs[3]:
    st.subheader(f"All metric × linkage combinations (k={K})")
    st.dataframe(res, use_container_width=True)
    st.caption("Higher is better: Silhouette, Calinski-Harabasz, Cophenetic. Lower is better: Davies-Bouldin.")
    ok = res.dropna(subset=["Silhouette"])
    if not ok.empty:
        best = ok.sort_values("Silhouette", ascending=False).iloc[0]
        st.success(f"Best silhouette: **{best.Metric} / {best.Linkage}** ({best.Silhouette})")
    st.subheader(f"Selected: {metric} / {linkage_choice}")
    st.write("Cluster sizes:", np.bincount(chosen_lab, minlength=K).tolist())

# ---- 5. Dendrograms
with tabs[4]:
    st.subheader("Linkages (Euclidean)")
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, link in zip(axes.ravel(), cl.LINKAGES):
        Z = Zs[("euclidean", link)]
        dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, no_labels=True, show_contracted=True)
        if link == "ward":
            ax.axhline(y=Z[-K + 1, 2] - 1e-6, color="r", ls="--", label=f"cut for k={K}")
            ax.legend()
        ax.set_title(f"{link.capitalize()} linkage (Euclidean)")
        ax.set_xlabel("Cluster (size)"); ax.set_ylabel("Distance")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.subheader("Average linkage across metrics")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for ax, m in zip(axes, cl.METRICS):
        dendrogram(Zs[(m, "average")], truncate_mode="lastp", p=30, ax=ax, no_labels=True)
        ax.set_title(f"Average linkage ({m})")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    Zw = Zs[("euclidean", "ward")]
    st.subheader("Ward merge heights")
    st.write("Last 8 merge heights:", np.round(Zw[-8:, 2], 2).tolist())
    st.write("Gaps between successive heights (largest gap suggests the cut):",
             np.round(np.diff(Zw[-8:, 2]), 2).tolist())

# ---- 6. Divisive vs Ward
with tabs[5]:
    from sklearn.metrics import adjusted_rand_score
    rows = [{"Method": "Divisive (Bisecting K-Means)", **cl.scores(Xs, div_lab),
             "Sizes": np.bincount(div_lab).tolist()},
            {"Method": "Agglomerative (Ward)", **cl.scores(Xs, ward_lab),
             "Sizes": np.bincount(ward_lab).tolist()}]
    st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("ARI: Divisive vs Ward", round(adjusted_rand_score(div_lab, ward_lab), 3))
    c2.metric("ARI: Divisive vs Average", round(adjusted_rand_score(div_lab, labels[("euclidean", "average")]), 3))

# ---- 7. Explore k
with tabs[6]:
    exp = _explore(Xs, seed)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, g in exp.groupby("Method"):
        ax[0].plot(g.k, g.Silhouette, marker="o", label=name)
        ax[1].plot(g.k, g.CH, marker="o", label=name)
    ax[0].set_title("Silhouette vs k"); ax[1].set_title("Calinski-Harabasz vs k")
    for a in ax:
        a.set_xlabel("k"); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.dataframe(exp.round(3), use_container_width=True)
    st.subheader("Ward cluster sizes as k increases (nested splitting)")
    Zw = Zs[("euclidean", "ward")]
    for k in range(2, 7):
        l = fcluster(Zw, k, criterion="maxclust")
        st.write(f"k={k}:", sorted(np.bincount(l)[1:].tolist(), reverse=True))

# ---- 8. PCA
with tabs[7]:
    P, evr, loadings = cl.pca_2d(Xs, features)
    st.write(f"Explained variance: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, total={evr.sum():.3f}")
    st.dataframe(loadings, use_container_width=False)
    panels = [("Single", labels[("euclidean", "single")]),
              ("Complete", labels[("euclidean", "complete")]),
              ("Average", labels[("euclidean", "average")]),
              ("Ward", ward_lab), ("Divisive (Bisecting)", div_lab)]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (name, lab) in zip(axes.ravel(), panels):
        ax.scatter(P[:, 0], P[:, 1], c=lab, cmap="viridis", s=18, alpha=.8)
        ax.set_title(f"{name} (k={K})"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax = axes.ravel()[-1]
    if "Channel" in meta.columns:
        ax.scatter(P[:, 0], P[:, 1], c=meta["Channel"], cmap="coolwarm", s=18, alpha=.8)
        ax.set_title("Actual Channel (reference)")
    else:
        ax.axis("off")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.subheader(f"Your selection: {metric} / {linkage_choice}")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(P[:, 0], P[:, 1], c=chosen_lab, cmap="viridis", s=22, alpha=.8)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    st.pyplot(fig); plt.close(fig)

# ---- 9. ARI
with tabs[8]:
    ari = cl.ari_matrix(labels, div_lab)
    st.write("Adjusted Rand Index between configurations")
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(ari.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ari))); ax.set_xticklabels(ari.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(ari))); ax.set_yticklabels(ari.index)
    for i in range(len(ari)):
        for j in range(len(ari)):
            ax.text(j, i, f"{ari.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.dataframe(ari, use_container_width=True)

# ---- 10. Profiles
with tabs[9]:
    which = st.radio("Method", ["Ward", "Divisive", f"Selected ({metric}/{linkage_choice})"], horizontal=True)
    lab = {"Ward": ward_lab, "Divisive": div_lab}.get(which, chosen_lab)
    st.subheader(f"Mean annual spend per cluster (k={K})")
    out = cl.profile(X, meta, lab)
    st.dataframe(out, use_container_width=True)
    st.bar_chart(out[features])
    for kk in (2, 4):
        l = fcluster(Zs[("euclidean", "ward")], kk, criterion="maxclust") - 1
        with st.expander(f"Ward k={kk} profile"):
            o = X.copy(); o["Cluster"] = l
            o = o.groupby("Cluster").mean().round(0); o["Size"] = pd.Series(l).value_counts().sort_index()
            st.dataframe(o, use_container_width=True)
    dl = df.copy(); dl["Cluster"] = lab
    st.download_button("⬇️ Download data with cluster labels", dl.to_csv(index=False).encode(),
                       "wholesale_clustered.csv", "text/csv")
