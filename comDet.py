import time
from collections import deque, defaultdict

# Zachary's Karate Club Dataset
KARATE_CLUB_EDGES = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (0, 13),
    (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21),
    (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
    (3, 7), (3, 12), (3, 13),
    (4, 6), (4, 10),
    (5, 6), (5, 10), (5, 16),
    (6, 16),
    (8, 30), (8, 32), (8, 33),
    (9, 33),
    (13, 33),
    (14, 32), (14, 33),
    (15, 32), (15, 33),
    (18, 32), (18, 33),
    (19, 33),
    (20, 32), (20, 33),
    (22, 32), (22, 33),
    (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
    (24, 25), (24, 27), (24, 31),
    (25, 31),
    (26, 29), (26, 33),
    (27, 33),
    (28, 31), (28, 33),
    (29, 32), (29, 33),
    (30, 32), (30, 33),
    (31, 32), (31, 33),
    (32, 33)
]

NUM_NODES = 34


class Graph:
    """Klasa za reprezentaciju grafa"""
    
    def __init__(self, num_nodes, edges):
        self.num_nodes = num_nodes
        self.edges = edges
        self.adj_list = self._create_adjacency_list()
        self.degrees = self._calculate_degrees()
        
    def _create_adjacency_list(self):
        """Kreira listu susedstva"""
        adj_list = [[] for _ in range(self.num_nodes)]
        for u, v in self.edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        return adj_list
    
    def _calculate_degrees(self):
        """Računa stepene čvorova"""
        return [len(neighbors) for neighbors in self.adj_list]
    
    def get_num_edges(self):
        """Vraća broj ivica"""
        return len(self.edges)


class LouvainAlgorithm:
    """Implementacija Louvain algoritma za detekciju zajednica"""
    
    def __init__(self, graph):
        self.graph = graph
        self.communities = list(range(graph.num_nodes))
        
    def calculate_modularity(self, communities=None):
        """Računa modularnost Q"""
        if communities is None:
            communities = self.communities
            
        m = self.graph.get_num_edges()
        Q = 0.0
        
        for i in range(self.graph.num_nodes):
            for j in range(self.graph.num_nodes):
                if communities[i] == communities[j]:
                    # Proveri da li postoji ivica između i i j
                    A_ij = 1 if j in self.graph.adj_list[i] else 0
                    ki = self.graph.degrees[i]
                    kj = self.graph.degrees[j]
                    Q += A_ij - (ki * kj) / (2.0 * m)
        
        return Q / (2.0 * m)
    
    def calculate_modularity_gain(self, node, from_comm, to_comm):
        """Računa dobitak u modularnosti pri pomeranju čvora"""
        m = self.graph.get_num_edges()
        
        # Broj ivica od node ka from_comm i to_comm
        ki_in_from = sum(1 for neighbor in self.graph.adj_list[node] 
                         if self.communities[neighbor] == from_comm)
        ki_in_to = sum(1 for neighbor in self.graph.adj_list[node] 
                       if self.communities[neighbor] == to_comm)
        
        ki = self.graph.degrees[node]
        delta_Q = (ki_in_to - ki_in_from) / (2.0 * m)
        
        return delta_Q
    
    def run(self, max_iterations=50):
        """Izvršava Louvain algoritam"""
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for node in range(self.graph.num_nodes):
                current_community = self.communities[node]
                
                # Pronađi susedne zajednice
                neighbor_communities = set()
                for neighbor in self.graph.adj_list[node]:
                    neighbor_communities.add(self.communities[neighbor])
                
                best_community = current_community
                best_gain = 0.0
                
                # Isprobaj pomeranje u svaku susednu zajednicu
                for community in neighbor_communities:
                    if community == current_community:
                        continue
                    
                    gain = self.calculate_modularity_gain(node, current_community, community)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_community = community
                
                # Pomeri čvor ako poboljšava modularnost
                if best_community != current_community:
                    self.communities[node] = best_community
                    improved = True
        
        return self._renumber_communities()
    
    def _renumber_communities(self):
        """Renumerише zajednice od 0 do k-1"""
        mapping = {}
        next_id = 0
        result = []
        
        for comm in self.communities:
            if comm not in mapping:
                mapping[comm] = next_id
                next_id += 1
            result.append(mapping[comm])
        
        return result


class GirvanNewmanAlgorithm:
    """Implementacija Girvan-Newman algoritma za detekciju zajednica"""
    
    def __init__(self, graph):
        self.graph = graph
        self.edges = list(graph.edges)
        
    def bfs_shortest_paths(self, source):
        """BFS za računanje najkraćih putanja"""
        distances = [-1] * self.graph.num_nodes
        num_paths = [0] * self.graph.num_nodes
        predecessors = [[] for _ in range(self.graph.num_nodes)]
        
        distances[source] = 0
        num_paths[source] = 1
        
        queue = deque([source])
        order = []
        
        while queue:
            v = queue.popleft()
            order.append(v)
            
            for neighbor in self.graph.adj_list[v]:
                # Pronađi suseda
                if distances[neighbor] < 0:
                    queue.append(neighbor)
                    distances[neighbor] = distances[v] + 1
                
                # Ako je na najkraćoj putanji
                if distances[neighbor] == distances[v] + 1:
                    num_paths[neighbor] += num_paths[v]
                    predecessors[neighbor].append(v)
        
        return order, num_paths, predecessors
    
    def calculate_edge_betweenness(self):
        """Računa edge betweenness za sve ivice"""
        betweenness = defaultdict(float)
        
        # Inicijalizuj betweenness za sve ivice
        for u, v in self.edges:
            key = (min(u, v), max(u, v))
            betweenness[key] = 0.0
        
        # Računaj betweenness iz svakog čvora kao izvora
        for source in range(self.graph.num_nodes):
            order, num_paths, predecessors = self.bfs_shortest_paths(source)
            
            # Dependency accumulation
            dependency = [0.0] * self.graph.num_nodes
            
            # Idi unazad kroz čvorove
            while order:
                w = order.pop()
                
                for v in predecessors[w]:
                    # Računaj koliko putanja ide preko ivice (v, w)
                    if num_paths[w] > 0:
                        flow = (num_paths[v] / num_paths[w]) * (1.0 + dependency[w])
                        dependency[v] += flow
                        
                        # Ažuriraj betweenness za ivicu
                        key = (min(v, w), max(v, w))
                        if key in betweenness:
                            betweenness[key] += flow
        
        return betweenness
    
    def remove_edge(self, edge_to_remove):
        """Uklanja ivicu iz grafa"""
        self.edges = [e for e in self.edges if not (
            (e[0] == edge_to_remove[0] and e[1] == edge_to_remove[1]) or
            (e[0] == edge_to_remove[1] and e[1] == edge_to_remove[0])
        )]
        
        # Ažuriraj adjacency listu
        u, v = edge_to_remove
        if v in self.graph.adj_list[u]:
            self.graph.adj_list[u].remove(v)
        if u in self.graph.adj_list[v]:
            self.graph.adj_list[v].remove(u)
    
    def find_connected_components(self):
        """Pronalazi povezane komponente (zajednice)"""
        visited = [False] * self.graph.num_nodes
        components = [-1] * self.graph.num_nodes
        component_id = 0
        
        for start in range(self.graph.num_nodes):
            if not visited[start]:
                # BFS za pronalaženje komponente
                queue = deque([start])
                visited[start] = True
                components[start] = component_id
                
                while queue:
                    node = queue.popleft()
                    for neighbor in self.graph.adj_list[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            components[neighbor] = component_id
                            queue.append(neighbor)
                
                component_id += 1
        
        return components
    
    def get_num_communities(self, communities):
        """Vraća broj različitih zajednica"""
        return len(set(communities))
    
    def run(self, target_communities=2):
        """Izvršava Girvan-Newman algoritam"""
        communities = list(range(self.graph.num_nodes))
        
        while self.get_num_communities(communities) < target_communities and self.edges:
            # Računaj edge betweenness
            betweenness = self.calculate_edge_betweenness()
            
            if not betweenness:
                break
            
            # Pronađi ivicu sa maksimalnom betweenness
            max_edge = max(betweenness.items(), key=lambda x: x[1])
            edge_to_remove = max_edge[0]
            
            # Ukloni ivicu
            self.remove_edge(edge_to_remove)
            
            # Pronađi nove zajednice
            communities = self.find_connected_components()
        
        return communities


def analyze_communities(communities):
    """Analizira rezultate detekcije zajednica"""
    community_dict = defaultdict(list)
    for node, comm in enumerate(communities):
        community_dict[comm].append(node)
    
    return dict(community_dict)


def print_results(algorithm_name, communities, modularity, execution_time):
    """Ispisuje rezultate algoritma"""
    print(f"\n{'='*60}")
    print(f"{algorithm_name}")
    print(f"{'='*60}")
    
    community_analysis = analyze_communities(communities)
    num_communities = len(community_analysis)
    
    print(f"Broj detektovanih zajednica: {num_communities}")
    print(f"Modularnost (Q): {modularity:.4f}")
    print(f"Vreme izvršavanja: {execution_time*1000:.2f} ms")
    
    print(f"\nDistribucija čvorova po zajednicama:")
    for comm_id, nodes in sorted(community_analysis.items()):
        print(f"  Zajednica {comm_id + 1}: {len(nodes)} čvorova - {nodes}")


def compare_algorithms(louvain_result, newman_result):
    """Poredi rezultate dva algoritma"""
    print(f"\n{'='*60}")
    print("POREĐENJE ALGORITAMA")
    print(f"{'='*60}")
    
    print(f"\n{'Metrika':<25} {'Louvain':<15} {'Girvan-Newman':<15}")
    print(f"{'-'*55}")
    print(f"{'Modularnost (Q)':<25} {louvain_result['modularity']:<15.4f} {newman_result['modularity']:<15.4f}")
    print(f"{'Broj zajednica':<25} {louvain_result['num_communities']:<15} {newman_result['num_communities']:<15}")
    print(f"{'Vreme (ms)':<25} {louvain_result['time']*1000:<15.2f} {newman_result['time']*1000:<15.2f}")
    
    # Najbolji algoritam po modularnosti
    if louvain_result['modularity'] > newman_result['modularity']:
        best = "Louvain"
        diff = louvain_result['modularity'] - newman_result['modularity']
    else:
        best = "Girvan-Newman"
        diff = newman_result['modularity'] - louvain_result['modularity']
    
    print(f"\n{'='*60}")
    print(f"ZAKLJUČAK:")
    print(f"{'='*60}")
    print(f"Algoritam sa boljom modularnošću: {best}")
    print(f"Razlika u modularnosti: {diff:.4f}")
    
    if louvain_result['time'] < newman_result['time']:
        faster = "Louvain"
        time_diff = newman_result['time'] - louvain_result['time']
    else:
        faster = "Girvan-Newman"
        time_diff = louvain_result['time'] - newman_result['time']
    
    print(f"Brži algoritam: {faster} (za {time_diff*1000:.2f} ms)")


def main():
    """Glavna funkcija"""
    print("="*60)
    print("COMMUNITY DETECTION U MREŽAMA")
    print("Dataset: Zachary's Karate Club")
    print("="*60)
    print(f"Broj čvorova: {NUM_NODES}")
    print(f"Broj ivica: {len(KARATE_CLUB_EDGES)}")
    
    # Kreiraj graf
    graph = Graph(NUM_NODES, KARATE_CLUB_EDGES)
    
    # ===== LOUVAIN ALGORITAM =====
    print("\n\n>>> Izvršavam LOUVAIN algoritam...")
    louvain = LouvainAlgorithm(graph)
    
    start_time = time.time()
    louvain_communities = louvain.run()
    louvain_time = time.time() - start_time
    
    louvain_modularity = louvain.calculate_modularity(louvain_communities)
    louvain_result = {
        'communities': louvain_communities,
        'modularity': louvain_modularity,
        'time': louvain_time,
        'num_communities': len(set(louvain_communities))
    }
    
    print_results("LOUVAIN ALGORITAM", louvain_communities, louvain_modularity, louvain_time)
    
    # ===== GIRVAN-NEWMAN ALGORITAM =====
    print("\n\n>>> Izvršavam GIRVAN-NEWMAN algoritam...")
    # Kreiraj novi graf jer Newman menja graf
    graph_newman = Graph(NUM_NODES, KARATE_CLUB_EDGES)
    newman = GirvanNewmanAlgorithm(graph_newman)
    
    start_time = time.time()
    newman_communities = newman.run(target_communities=2)
    newman_time = time.time() - start_time
    
    # Računaj modularnost za Newman rezultat
    louvain_temp = LouvainAlgorithm(Graph(NUM_NODES, KARATE_CLUB_EDGES))
    newman_modularity = louvain_temp.calculate_modularity(newman_communities)
    
    newman_result = {
        'communities': newman_communities,
        'modularity': newman_modularity,
        'time': newman_time,
        'num_communities': len(set(newman_communities))
    }
    
    print_results("GIRVAN-NEWMAN ALGORITAM", newman_communities, newman_modularity, newman_time)
    
    # ===== POREĐENJE =====
    compare_algorithms(louvain_result, newman_result)


if __name__ == "__main__":
    main()