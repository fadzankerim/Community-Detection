import time
import copy
import random
from collections import deque, defaultdict


NUM_NODES = 300
NUM_COMMUNITIES = 5
NODES_PER_COMMUNITY = NUM_NODES // NUM_COMMUNITIES
# Gustina povezivanja
INTERNAL_DENSITY = 0.8
EXTERNAL_DENSITY = 0.02


def generate_large_simulated_graph(num_nodes, num_communities, internal_density, external_density):
    
    print(f"Generisanje simuliranog grafa: {num_nodes} čvorova, {num_communities} zajednica...")
    all_edges = set()
    
    # 1. Generisanje gustih veza unutar zajednica
    for comm_id in range(num_communities):
        start_node = comm_id * NODES_PER_COMMUNITY
        end_node = start_node + NODES_PER_COMMUNITY
        community_nodes = list(range(start_node, end_node))
        
        for i in range(len(community_nodes)):
            for j in range(i + 1, len(community_nodes)):
                u, v = community_nodes[i], community_nodes[j]
                if random.random() < internal_density:
                    all_edges.add(tuple(sorted((u, v))))
                    
    # Generisanje rijetkih veza između zajednica
    for comm_a in range(num_communities):
        for comm_b in range(comm_a + 1, num_communities):
            start_a = comm_a * NODES_PER_COMMUNITY
            end_a = start_a + NODES_PER_COMMUNITY
            
            start_b = comm_b * NODES_PER_COMMUNITY
            end_b = start_b + NODES_PER_COMMUNITY
            
            for u in range(start_a, end_a):
                for v in range(start_b, end_b):
                    if random.random() < external_density:
                        all_edges.add(tuple(sorted((u, v))))

    print(f"Završeno. Ukupno bridova (simuliranih): {len(all_edges)}")
    return list(all_edges)

def load_edges_from_csv(filepath):
  
    
    print(f"\n[UPOZORENJE]: Čitanje iz CSV datoteke '{filepath}' je simulirano.")
    print("Koristimo generisani graf od 300 čvorova za testiranje skalabilnosti.\n")

    global EXPANDED_EDGES, NUM_NODES
    EXPANDED_EDGES = generate_large_simulated_graph(NUM_NODES, NUM_COMMUNITIES, INTERNAL_DENSITY, EXTERNAL_DENSITY)
    # Postavljamo NUM_NODES na najveći ID čvora + 1
    max_node_id = 0
    for u, v in EXPANDED_EDGES:
        max_node_id = max(max_node_id, u, v)
    NUM_NODES = max_node_id + 1
    
    return EXPANDED_EDGES, NUM_NODES

EXPANDED_EDGES, NUM_NODES = load_edges_from_csv("veliki_socijalni_graf.csv")


class Graph:
    
    
    def __init__(self, num_nodes, edges):
        self.num_nodes = num_nodes
        self.edges = list(edges) # Čuvamo kopiju ivica
        self.adj_list = self._create_adjacency_list()
        self.degrees = self._calculate_degrees()
        
    def _create_adjacency_list(self):
        #lista susjedstva
        adj_list = [[] for _ in range(self.num_nodes)]
        for u, v in self.edges:
            # Osiguravamo da su čvorovi unutar opsega NUM_NODES
            if u < self.num_nodes and v < self.num_nodes:
                adj_list[u].append(v)
                adj_list[v].append(u)
        return adj_list
    
    def _calculate_degrees(self):
        """Računa stepene (stupnjeve) čvorova"""
        return [len(neighbors) for neighbors in self.adj_list]
    
    def get_num_edges(self):
        """Vraća broj bridova (ivica)"""
        unique_edges = set()
        for u, v in self.edges:
            unique_edges.add(tuple(sorted((u, v))))
        return len(unique_edges)


class LouvainAlgorithm:
    """Implementacija Louvain algoritma za detekciju zajednica"""
    
    def __init__(self, graph):
        self.graph = graph
        # U Louvain-u svaka čvor inicijalno je svoja zajednica
        self.communities = list(range(graph.num_nodes))
        
    def calculate_modularity(self, communities=None):
        """Računa modularnost Q za datu podjelu"""
        if communities is None:
            communities = self.communities
            
        m = self.graph.get_num_edges()
        if m == 0:
            return 0.0
            
        Q_total = 0.0
        
        for i in range(self.graph.num_nodes):
            for j in range(self.graph.num_nodes):
                if communities[i] == communities[j]:
                    # Provjerava postoji li ivica (A_ij = 1) - O(k_i) gdje je k_i stepen čvora i
                    A_ij = 1 if j in self.graph.adj_list[i] else 0
                    ki = self.graph.degrees[i]
                    kj = self.graph.degrees[j]
                    Q_total += A_ij - (ki * kj) / (2.0 * m)
        
        return Q_total / (2.0 * m)
    
    def calculate_modularity_gain(self, node, from_comm, to_comm):
        
        m = self.graph.get_num_edges()
        
        # Broj ivica od čvora ka from_comm
        k_i_in_from = sum(1 for neighbor in self.graph.adj_list[node] 
                         if self.communities[neighbor] == from_comm)
        
        # Broj ivica od čvora ka to_comm
        k_i_in_to = sum(1 for neighbor in self.graph.adj_list[node] 
                       if self.communities[neighbor] == to_comm)
        
        # Dobitak: (veze u novu zajednicu) - (veze iz stare zajednice)
        delta_Q = (k_i_in_to - k_i_in_from) / (2.0 * m)
        
        return delta_Q
    
    def run(self, max_iterations=50):
        
        improved = True
        iteration = 0
        
        # Inicijalno, svaka čvor je svoja zajednica
        self.communities = list(range(self.graph.num_nodes)) 
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Nasumična iteracija čvorova radi boljih rezultata
            node_order = list(range(self.graph.num_nodes))
            random.shuffle(node_order)
            
            for node in node_order:
                current_community = self.communities[node]
                
                # Pronađi susjedne zajednice (potencijalne destinacije)
                neighbor_communities = set()
                for neighbor in self.graph.adj_list[node]:
                    neighbor_communities.add(self.communities[neighbor])
                
                best_community = current_community
                best_gain = 0.0
                
                # Isprobaj pomjeranje u svaku susjednu zajednicu
                for community in neighbor_communities:
                    if community == current_community:
                        continue
                    
                    # Računaj dobitak - Brzi O(k_i) proračun
                    gain = self.calculate_modularity_gain(node, current_community, community)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_community = community
                
                # Pomjeri čvor ako poboljšava modularnost
                if best_community != current_community:
                    self.communities[node] = best_community
                    improved = True
        
        # Faza 2: Agregacija (nije implementirana ovdje, vraćamo samo Fazu 1 rezultate)
        return self._renumber_communities()
    
    def _renumber_communities(self):
        """Renumeriše zajednice od 0 do k-1 radi čitljivosti"""
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
    
    
    def __init__(self, graph):
        self.graph = graph
        # Koristimo kopije za GN jer modifikuje listu bridova
        self.current_edges = list(graph.edges)
        
    def _create_temp_adj_list(self):
        
        adj_list = [[] for _ in range(self.graph.num_nodes)]
        # Osiguravamo da se duplikati ne dodaju
        unique_edges = set(tuple(sorted(e)) for e in self.current_edges)

        for u, v in unique_edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        return adj_list
        
    def bfs_shortest_paths(self, source, adj_list):
        #bfs za  racucanje najkracih putanja
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
            
            for neighbor in adj_list[v]:
                # Pronađi susjeda
                if distances[neighbor] < 0:
                    queue.append(neighbor)
                    distances[neighbor] = distances[v] + 1
                
                # Ako je na najkraćoj putanji
                if distances[neighbor] == distances[v] + 1:
                    num_paths[neighbor] += num_paths[v]
                    predecessors[neighbor].append(v)
        
        return order, num_paths, predecessors
    
    def calculate_edge_betweenness(self, adj_list):
        
        betweenness = defaultdict(float)
        
        # Inicijaliziraj betweenness za sve jedinstvene ivice
        unique_edges = set(tuple(sorted(e)) for e in self.current_edges)
        for key in unique_edges:
            betweenness[key] = 0.0
        
        # Računaj betweenness iz svakog čvora kao izvora
        for source in range(self.graph.num_nodes):
            order, num_paths, predecessors = self.bfs_shortest_paths(source, adj_list)
            
            # Akumulacija zavisnosti
            dependency = [0.0] * self.graph.num_nodes
            
            # Idi unazad kroz čvorove
            # Uklanjamo dependency s cilja prema izvoru
            for w in reversed(order):
                if w == source:
                    continue
                    
                for v in predecessors[w]:
                    # Računaj koliko putanja ide preko ivice (v, w)
                    if num_paths[w] > 0:
                        flow = (num_paths[v] / num_paths[w]) * (1.0 + dependency[w])
                        dependency[v] += flow
                        
                        # Ažuriraj betweenness za ivice
                        key = tuple(sorted((v, w)))
                        if key in betweenness:
                            betweenness[key] += flow
        
        for key in betweenness:
            betweenness[key] /= 2.0
            
        return betweenness
    
    def remove_edge(self, edge_to_remove):
        #uklanja ivice iz grafa
        u, v = edge_to_remove
        
        # Filtriraj sve instance tog brida (zbog mogućeg dupliranja u listi)
        self.current_edges = [e for e in self.current_edges if not (
            (e[0] == u and e[1] == v) or
            (e[0] == v and e[1] == u)
        )]
    
    def find_connected_components(self, adj_list):
        """Pronalazi povezane komponente (zajednice) pomoću BFS-a na trenutnom grafu"""
        visited = [False] * self.graph.num_nodes
        components = [-1] * self.graph.num_nodes
        component_id = 0
        
        for start in range(self.graph.num_nodes):
            if not visited[start]:
                queue = deque([start])
                visited[start] = True
                components[start] = component_id
                
                while queue:
                    node = queue.popleft()
                    for neighbor in adj_list[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            components[neighbor] = component_id
                            queue.append(neighbor)
                
                component_id += 1
        
        return components
    
    def run(self):
        
        best_modularity = -1.0
        best_communities = list(range(self.graph.num_nodes)) 
        
        # Kreiramo objekat za računanje modularnosti na originalnom grafu (za konzistentnost)
        modularity_calculator = LouvainAlgorithm(Graph(self.graph.num_nodes, EXPANDED_EDGES)) 
        
        # Glavna petlja: iterativno uklanjamo ivice dok god ih ima
        while self.current_edges:
            
            # 1. Kreiraj listu susjedstva za trenutno stanje
            adj_list = self._create_temp_adj_list()
            
            # 2. Pronađi nove zajednice (povezane komponente)
            current_communities = self.find_connected_components(adj_list)
            
            # 3. Računaj modularnost za trenutnu podjelu
            current_modularity = modularity_calculator.calculate_modularity(current_communities)
            
            # 4. Ažuriraj najbolji rezultat
            if current_modularity > best_modularity:
                best_modularity = current_modularity
                best_communities = copy.deepcopy(current_communities) 
            
            if not self.current_edges:
                break
                
            # 5. Računaj edge betweenness
            betweenness = self.calculate_edge_betweenness(adj_list)
            
            if not betweenness:
                break
            
            # 6. Pronađi brid sa maksimalnom betweenness
            max_edge_key = max(betweenness.items(), key=lambda x: x[1])[0]
            
            # 7. Ukloni brid
            self.remove_edge(max_edge_key)

        return best_communities, best_modularity 


def analyze_communities(communities):
    """Analizira rezultate detekcije zajednica"""
    community_dict = defaultdict(list)
    for node, comm in enumerate(communities):
        community_dict[comm].append(node)
    
    # Renumiramo ID-eve da krenu od 0, 1, 2...
    mapping = {comm_id: i for i, comm_id in enumerate(sorted(community_dict.keys()))}
    final_dict = {mapping[comm_id]: nodes for comm_id, nodes in community_dict.items()}
    
    return final_dict


def print_results(algorithm_name, communities, modularity, execution_time):
    """Ispisuje rezultate algoritma"""
    print(f"\n{'='*60}")
    print(f"{algorithm_name}")
    print(f"{'='*60}")
    
    community_analysis = analyze_communities(communities)
    num_communities = len(community_analysis)
    
    print(f"Broj detektovanih zajednica: {num_communities}")
    print(f"Modularnost (Q): {modularity:.4f}")
    print(f"Vrijeme izvršavanja: {execution_time*1000:.2f} ms")
    
    print(f"\nDistribucija čvorova po zajednicama:")
    # Printamo samo prvih 5 zajednica radi čitljivosti
    for i, (comm_id, nodes) in enumerate(sorted(community_analysis.items())):
        if i >= 5:
            print(f"  ... i još {len(community_analysis) - 5} zajednica.")
            break
        node_preview = nodes[:10]
        if len(nodes) > 10:
             node_preview.append('...')
        print(f"  Zajednica {comm_id + 1}: {len(nodes)} čvorova - {node_preview}")


def compare_algorithms(louvain_result, newman_result):
    """Poredi rezultate dva algoritma"""
    print(f"\n{'='*60}")
    print("POREĐENJE ALGORITAMA (VELIKI SIMULIRANI DATASET)")
    print(f"{'='*60}")
    
    print(f"\n{'Metrika':<25} {'Louvain':<15} {'Girvan-Newman':<15}")
    print(f"{'-'*55}")
    print(f"{'Modularnost (Q)':<25} {louvain_result['modularity']:<15.4f} {newman_result['modularity']:<15.4f}")
    print(f"{'Broj zajednica':<25} {louvain_result['num_communities']:<15} {newman_result['num_communities']:<15}")
    # Ispisujemo vrijeme u sekundama (s) jer je Girvan-Newman mnogo sporiji
    print(f"{'Vrijeme (s)':<25} {louvain_result['time']:<15.3f} {newman_result['time']:<15.3f}")
    
    # Zaključak
    print(f"\n{'='*60}")
    print(f"ZAKLJUČAK O SKALABILNOSTI:")
    print(f"{'='*60}")
    
    if louvain_result['time'] > 0 and newman_result['time'] > 0:
        time_ratio = newman_result['time'] / louvain_result['time']
        print(f"Na grafu od {NUM_NODES} čvorova i {len(EXPANDED_EDGES)} bridova:")
        print(f"Girvan-Newman je otprilike {time_ratio:.1f} puta sporiji od Louvaina.")
    
    print("Louvain ostaje superioran za velike mreže zbog svoje linearne vremenske složenosti.")


def main():
    """Glavna funkcija za izvršavanje i poređenje algoritama"""
    print("="*60)
    print("DETEKCIJA ZAJEDNICA U MREŽAMA")
    print(f"Dataset: VELIKI SIMULIRANI GRAF ({NUM_NODES} čvorova, {len(EXPANDED_EDGES)} bridova)")
    print("="*60)
    
    initial_graph = Graph(NUM_NODES, EXPANDED_EDGES)
    
    #  LOUVAIN ALGORITAM 
    print("\n\n>>> Izvršavam LOUVAIN algoritam...")
    louvain_graph = Graph(NUM_NODES, EXPANDED_EDGES)
    louvain = LouvainAlgorithm(louvain_graph)
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
    
    #  GIRVAN-NEWMAN ALGORITAM 
    print("\n\n>>> Izvršavam GIRVAN-NEWMAN algoritam (Ovo će trajati duže)...")
    newman_graph = Graph(NUM_NODES, EXPANDED_EDGES)
    newman = GirvanNewmanAlgorithm(newman_graph)
    
    start_time = time.time()
    newman_communities, newman_modularity = newman.run()
    newman_time = time.time() - start_time
    
    newman_result = {
        'communities': newman_communities,
        'modularity': newman_modularity,
        'time': newman_time,
        'num_communities': len(set(newman_communities))
    }
    
    print_results("GIRVAN-NEWMAN ALGORITAM (Optimalna Q)", newman_communities, newman_modularity, newman_time)
    
    #  POREĐENJE 
    compare_algorithms(louvain_result, newman_result)


if __name__ == "__main__":
    main()