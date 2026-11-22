### APPLIED NETWORK SCIENCE -- SHARED HELPERS (as of week 06)
### functions built together in practice sessions, collected here for reuse.
### each helper below is tagged with the week/section that introduced it.
### (week 06 -- degree-preserving null models -- introduces no new helpers: every step is
### written inline in the notebook, nothing to carry)

import numpy as np
import pandas as pd
import networkx as nx
from infomap import Infomap


### week 01 -- introduction
def graph_summary(G):
    """Print a quick: directed/weighted flags, node/edge counts, density, and components."""
    print("Directed:", G.is_directed())
    print("Weighted:", nx.is_weighted(G))
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("Density:", nx.density(G))
    if G.is_directed():
        print("Number of weakly connected components:", nx.number_weakly_connected_components(G))
    else:
        print("Number of components:", nx.number_connected_components(G))


### week 03 -- reusable helper functions
def giant_metrics(G):
    """Structural profile of a network, with path length and diameter measured on the giant component."""
    giant_nodes = max(nx.connected_components(G), key=len)
    gc = G.subgraph(giant_nodes)
    return {
        "nr_nodes": G.number_of_nodes(),
        "nr_edges": G.number_of_edges(),
        "nr_nodes_giant": gc.number_of_nodes(),
        "nr_edges_giant": gc.number_of_edges(),
        "avg_degree": 2 * G.number_of_edges() / G.number_of_nodes(),
        "clustering": nx.transitivity(G),
        "avg_path_len_giant": nx.average_shortest_path_length(gc),
        "diameter_giant": nx.diameter(gc),
        "components": nx.number_connected_components(G),
    }


### week 02, section 02 -- Clustering and average path length in random and small-world networks
def average_metrics(graphs):
    """Run giant_metrics on a list of graphs and average degree/clustering/path length across them
    -- the same helper is used for both the random and the Watts-Strogatz summaries."""
    df = pd.DataFrame([giant_metrics(G) for G in graphs])
    return {
        "avg_degree": df["avg_degree"].mean(),
        "clustering": df["clustering"].mean(),
        "avg_path_length": df["avg_path_len_giant"].mean(),
    }


### week 03, section 01 -- Scale-free vs random networks
def degree_pk(G):
    """Empirical degree distribution P(k): how many nodes have each degree, normalized to sum to 1."""
    degrees = pd.Series([d for _, d in G.degree()])
    counts = degrees.value_counts().sort_index()
    return pd.DataFrame({
        "degree": counts.index,
        "count": counts.values,
        "pk": counts.values / counts.values.sum(),
    })


### week 03, section 02 -- Different approaches to plotting degree distributions
def log_binned_pdf(G, n_bins=15):
    """Degree distribution grouped into logarithmically spaced bins, so the sparse, noisy tail
    gets pooled into bins with more nodes each."""
    degrees = np.array([d for _, d in G.degree() if d > 0])

    edges = np.unique(np.floor(np.geomspace(degrees.min(), degrees.max() + 1, n_bins + 1)))
    if len(edges) < 3:
        edges = np.unique([degrees.min(), degrees.max(), degrees.max() + 1])

    counts, edges = np.histogram(degrees, bins=edges)
    left, right = edges[:-1], edges[1:]
    width = right - left
    mid = np.sqrt(left * right)  # geometric midpoint of each bin
    density = counts / (len(degrees) * width)  # probability per unit k
    return pd.DataFrame({"k": mid, "pk": density})


def ccdf_deg(G):
    """Fraction of nodes with degree >= k, for every observed positive degree -- standard in
    heavy-tail analysis, much less noisy in the tail than the raw PDF."""
    degrees = np.array([d for _, d in G.degree() if d > 0])
    ks = np.sort(np.unique(degrees))
    ccdf = [np.mean(degrees >= x) for x in ks]
    return pd.DataFrame({"k": ks, "ccdf": ccdf})


### week 03, section 03 -- Robustness of scale-free networks -- terrorist network
def random_attack(G):
    """Remove one uniformly random node."""
    node = np.random.choice(G.nodes())
    H = G.copy()
    H.remove_node(node)
    return H


def target_attack(G):
    """Remove the current highest-degree node."""
    node = max(G.degree(), key=lambda x: x[1])[0]
    H = G.copy()
    H.remove_node(node)
    return H


def gc_share(G):
    """Share of nodes still in the giant connected component."""
    return len(max(nx.connected_components(G), key=len)) / G.number_of_nodes()


### week 05, section 02 -- Infomap
def run_infomap(G, nb_trials=20, seed=None):
    """Run Infomap on G and return a {node: module_id} membership dict."""
    # --two-level: stop at flat communities, don't build a hierarchy of nested modules
    # --num-trials: Infomap's search is randomized, so it reruns nb_trials times internally
    # and keeps the partition with the shortest description length (the best of them)
    flags = f"--two-level --silent --num-trials {nb_trials}"
    if seed is not None:
        flags += f" --seed {seed}"
    im = Infomap(flags)

    # unlike the raw demo above, this needs to work for ANY node id, not just integers -- so
    # remap here, every time, regardless of what G's node ids look like
    node_to_id = {n: i for i, n in enumerate(G.nodes())}
    id_to_node = {i: n for n, i in node_to_id.items()}
    for u, v in G.edges():
        im.add_link(node_to_id[u], node_to_id[v])
    im.run()

    # im.nodes gives just the actual nodes with their module assignment
    return {id_to_node[node.node_id]: node.module_id for node in im.nodes}


def sets_from_membership(membership):
    """Turn a {node: community_id} dict into the list-of-sets format networkx expects."""
    # expected input: a dict like best_partition() returns -- one entry per node, keys are
    # node ids, values are their community id (int or any hashable, doesn't need to be 0-indexed)
    # invert the dict: group all nodes that share the same community_id into one set
    communities = {}
    for node, comm_id in membership.items():
        if comm_id not in communities:
            communities[comm_id] = set()
        communities[comm_id].add(node)
    return list(communities.values())


### week 05, section 03 -- Girvan-Newman algorithm
def girvan_newman_partitions(G, max_communities):
    """Repeatedly remove the highest-betweenness edge, splitting off one more community at a
    time, until we reach max_communities; return {n_communities: list_of_community_sets} along
    the way."""
    H = G.copy()  # work on a copy -- we're about to start deleting edges
    partitions = {1: [set(H.nodes())]}  # before any edge is removed, everyone is one community
    n_communities = 1

    while n_communities < max_communities and H.number_of_edges() > 0:
        # keep removing the top-betweenness edge until the network actually splits again --
        # most removals don't (only bridges do), exactly like the demo above
        while True:
            edge_betw = nx.edge_betweenness_centrality(H)
            top_edge = max(edge_betw, key=edge_betw.get)
            H.remove_edge(*top_edge)
            communities = list(nx.connected_components(H))
            if len(communities) > n_communities or H.number_of_edges() == 0:
                break

        n_communities = len(communities)
        partitions[n_communities] = communities

    return partitions


def membership_from_sets(communities):
    """Turn a list-of-sets partition into a {node: community_id} dict (the reverse of
    sets_from_membership)."""
    return {node: i for i, community in enumerate(communities) for node in community}
