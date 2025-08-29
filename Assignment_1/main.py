import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

# =========================
# Load only first N relationships from file
# =========================
N = 1000

with open("relationships.json") as f:
    data = json.load(f)

G = nx.DiGraph()

count = 0
for image in data:
    for rel in image['relationships']:
        if count >= N:
            break
        try:
            subj = rel['subject']['names'][0].lower()
        except KeyError:
            subj = rel['subject']['name'].lower()
        try:
            obj = rel['object']['names'][0].lower()
        except KeyError:
            obj = rel['object']['name'].lower()
        
        predicate = rel['predicate'].lower()
        G.add_edge(subj, obj, relation=predicate)
        count += 1
    if count >= N:
        break

# =========================
# DEGREE DISTRIBUTION
# =========================
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())
total_degrees = dict(G.degree())

avg_in = np.mean(list(in_degrees.values()))
avg_out = np.mean(list(out_degrees.values()))
avg_total = np.mean(list(total_degrees.values()))

print("Average in-degree:", avg_in)
print("Average out-degree:", avg_out)
print("Average total degree:", avg_total)

# Boxplots
plt.figure(figsize=(8,6))
plt.boxplot(
    [list(in_degrees.values()), 
     list(out_degrees.values()), 
     list(total_degrees.values())],
    tick_labels=["In-degree", "Out-degree", "Total degree"]
)
plt.ylabel("Degree")
plt.title(f"Degree Distribution of VG Graph (first {N} relationships)")
plt.show()

# Log-log plot of degree distribution
degrees = list(total_degrees.values())
plt.figure(figsize=(6,5))
counts, bins = np.histogram(degrees, bins=20)
plt.scatter(bins[:-1], counts, alpha=0.7)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Degree (log)")
plt.ylabel("Frequency (log)")
plt.title("Degree Distribution (log-log)")
plt.show()

# Identify outliers (top 1% nodes)
threshold = np.percentile(degrees, 99)
outliers = [n for n, d in total_degrees.items() if d >= threshold]
print("Outlier nodes:", outliers)

# =========================
# PATH LENGTHS
# =========================
if not nx.is_connected(G.to_undirected()):
    Gc = max(nx.connected_components(G.to_undirected()), key=len)
    G_sub = G.subgraph(Gc).copy()
else:
    G_sub = G

avg_path_len = nx.average_shortest_path_length(G_sub.to_undirected())
diameter = nx.diameter(G_sub.to_undirected())

print("Average path length:", avg_path_len)
print("Diameter:", diameter)
print("log(N):", math.log(len(G_sub)))

# find node pairs with diameter distance
diam_pairs = []
for u in G_sub.nodes():
    for v in G_sub.nodes():
        if u != v and nx.has_path(G_sub, u, v):
            if nx.shortest_path_length(G_sub, u, v) == diameter:
                diam_pairs.append((u, v))
print("Example node pairs with diameter:", diam_pairs[:5])

# =========================
# GEODESIC PATH LENGTH
# =========================
geo_avg = nx.average_shortest_path_length(G_sub.to_undirected())
print("Average geodesic path length (giant component):", geo_avg)

# =========================
# CLUSTERING COEFFICIENT
# =========================
clust_coeffs = nx.clustering(G_sub.to_undirected())
avg_cc = nx.average_clustering(G_sub.to_undirected())
print("Average clustering coefficient:", avg_cc)

deg_list = dict(G_sub.degree())
cc_values = [clust_coeffs[n] for n in G_sub.nodes()]
deg_values = [deg_list[n] for n in G_sub.nodes()]

plt.figure(figsize=(6,5))
plt.scatter(deg_values, cc_values, alpha=0.6)
plt.xscale('log')
plt.xlabel("Degree")
plt.ylabel("Clustering Coefficient")
plt.title("Clustering Coefficient vs Degree")
plt.show()

# =========================
# AVG CC by DEGREE GROUPS
# =========================
degrees_arr = np.array(list(deg_list.values()))
p25 = np.percentile(degrees_arr, 25)
p75 = np.percentile(degrees_arr, 75)

low_deg_nodes = [n for n, d in deg_list.items() if d <= p25]
high_deg_nodes = [n for n, d in deg_list.items() if d >= p75]

if low_deg_nodes:
    avg_cc_low = np.mean([clust_coeffs[n] for n in low_deg_nodes])
else:
    avg_cc_low = float('nan')

if high_deg_nodes:
    avg_cc_high = np.mean([clust_coeffs[n] for n in high_deg_nodes])
else:
    avg_cc_high = float('nan')

print("Avg CC (low-degree nodes):", avg_cc_low)
print("Avg CC (high-degree nodes):", avg_cc_high)

# =========================
# STRONGLY / WEAKLY CONNECTED COMPONENTS
# =========================
if G.is_directed():
    sccs = list(nx.strongly_connected_components(G))
    scc_sizes = sorted([len(c) for c in sccs], reverse=True)
    print("\nNumber of SCCs:", len(sccs))
    print("SCC size distribution (top 10):", scc_sizes[:10])

    wccs = list(nx.weakly_connected_components(G))
    wcc_sizes = sorted([len(c) for c in wccs], reverse=True)
    print("Number of WCCs:", len(wccs))
    print("WCC size distribution (top 10):", wcc_sizes[:10])
else:
    comps = list(nx.connected_components(G))
    comp_sizes = sorted([len(c) for c in comps], reverse=True)
    print("\nNumber of connected components:", len(comps))
    print("Component size distribution (top 10):", comp_sizes[:10])

# =========================
# GIANT COMPONENT & COVERAGE
# =========================
if G.is_directed():
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    G_giant = G.subgraph(largest_wcc).copy()
else:
    largest_cc = max(nx.connected_components(G), key=len)
    G_giant = G.subgraph(largest_cc).copy()

n_total, m_total = G.number_of_nodes(), G.number_of_edges()
n_giant, m_giant = G_giant.number_of_nodes(), G_giant.number_of_edges()

print("\n=== Giant Component ===")
print(f"Giant nodes: {n_giant}/{n_total} ({100*n_giant/n_total:.2f}%)")
print(f"Giant edges: {m_giant}/{m_total} ({100*m_giant/m_total:.2f}%)")

deg_all = np.mean([d for _, d in G.degree()]) if n_total > 0 else float('nan')
deg_giant = np.mean([d for _, d in G_giant.degree()]) if n_giant > 0 else float('nan')
cc_all = nx.average_clustering(G.to_undirected())
cc_giant = nx.average_clustering(G_giant.to_undirected())

print("Avg degree - full graph:", deg_all, "giant:", deg_giant)
print("Avg clustering - full graph:", cc_all, "giant:", cc_giant)

# =========================
# GIANT COMPONENT PATH METRICS + SMALL-WORLD CHECK
# =========================
Gu = G_giant.to_undirected()
n, m = Gu.number_of_nodes(), Gu.number_of_edges()

if n > 1 and nx.is_connected(Gu):
    giant_avg_path = nx.average_shortest_path_length(Gu)
    giant_diam = nx.diameter(Gu)
else:
    giant_avg_path = float('nan')
    giant_diam = float('nan')

giant_cc = nx.average_clustering(Gu)

print("\nGiant component avg path length:", giant_avg_path)
print("Giant component diameter:", giant_diam)
print("Giant component avg clustering:", giant_cc)

p = (2.0*m)/(n*(n-1)) if n > 1 else 0.0
expected_cc_random = p
avg_deg = np.mean([d for _, d in G_giant.degree()]) if n > 0 else float('nan')
approx_rand_path = math.log(n)/math.log(avg_deg) if n>1 and avg_deg>1 else float('nan')

print("Random-graph expected clustering:", expected_cc_random)
print("Approx random path length:", approx_rand_path)

if not math.isnan(giant_cc) and expected_cc_random > 0:
    if giant_cc > 5*expected_cc_random and not math.isnan(giant_avg_path) and not math.isnan(approx_rand_path):
        if giant_avg_path <= 2*approx_rand_path:
            print("Small-world features: LIKELY")
        else:
            print("Small-world features: UNCERTAIN")
    else:
        print("Small-world features: UNLIKELY")

# =========================
# K-CORE DECOMPOSITION (ROBUSTNESS)
# =========================
print("\n=== K-core Decomposition ===")

# remove self-loops, otherwise nx.k_core crashes
Gu.remove_edges_from(nx.selfloop_edges(Gu))

for k in range(1, 6):
    core = nx.k_core(Gu, k=k)
    print(f"k={k}: nodes={core.number_of_nodes()}, edges={core.number_of_edges()}")
