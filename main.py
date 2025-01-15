import os
import time
import tracemalloc
import numpy as np
import networkx as nx
from queue import PriorityQueue
from timeout_decorator import timeout, TimeoutError

TIME_LIMIT = 10  # Tempo limite de execução (segundos)

def read_tsp_file(path):
    """Lê coordenadas de arquivos TSPLIB."""
    coords = []
    parsing = False
    try:
        with open(path, 'r') as f:
            for line in f:
                l = line.strip()
                if l == 'NODE_COORD_SECTION':
                    parsing = True
                    continue
                if l == 'EOF':
                    break
                if parsing:
                    parts = l.split()
                    coords.append((float(parts[1]), float(parts[2])))
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {path}")
    except Exception as e:
        print(f"Erro ao ler arquivo {path}: {e}")
    return coords

def load_optimal_solutions(path):
    """Carrega soluções ótimas de um arquivo."""
    sols = {}
    if not os.path.isfile(path):
        print(f"Arquivo de soluções ótimas não encontrado: {path}")
        return sols
    with open(path, 'r') as f:
        for line in f:
            try:
                name, cost = line.strip().split(':')
                cost = ''.join(filter(str.isdigit, cost))  # Remove caracteres não numéricos
                if cost:
                    sols[name.strip()] = int(cost)
            except ValueError:
                print(f"Erro ao processar linha: {line.strip()}")
    return sols

def create_distance_matrix(coords):
    """Gera matriz de distâncias euclidianas."""
    n = len(coords)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            mat[i][j] = dist
            mat[j][i] = dist
    return mat

def measure_memory_and_time(func, *args, **kwargs):
    """Mede tempo e pico de memória durante o algoritmo.
    Retorna:
        - result: resultado da função
        - elapsed_time: tempo em segundos
        - peak_mem: pico de memória em KB (kilobytes)
    """
    tracemalloc.start()
    try:
        start_time = time.time()  # Tempo em segundos desde epoch
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time  # Duração em segundos
        current, peak_mem = tracemalloc.get_traced_memory()  # Memória em bytes
        return result, elapsed_time, peak_mem / 1024  # Converte bytes para KB
    finally:
        tracemalloc.stop()

@timeout(TIME_LIMIT)
def bnb_algorithm(mat):
    """Branch-and-Bound."""
    print("Executando Branch-and-Bound...")
    best_cost = float('inf')
    n = len(mat)
    
    # Pré-calcula as distâncias ordenadas para cada cidade
    sorted_distances = [sorted([mat[i][j] for j in range(n) if j != i]) for i in range(n)]
    best_edges = [row[0] + row[1] for row in sorted_distances]

    class Node:
        def __init__(self, path, cost, level):
            self.path = path
            self.cost = cost
            self.level = level
            self.bound = calculate_bound(mat, path, best_edges)

        def __lt__(self, other):
            return self.bound < other.bound

    def calculate_bound(mat, path, best_edges):
        bound = sum(best_edges[i] for i in range(len(mat)) if i not in path) / 2
        for i in range(len(path) - 1):
            bound += mat[path[i], path[i + 1]]
        return bound

    queue = PriorityQueue()
    queue.put(Node([0], 0, 0))

    while not queue.empty():
        current = queue.get()
        if current.bound >= best_cost:
            continue
        if current.level == n - 1:
            total_cost = current.cost + mat[current.path[-1], current.path[0]]
            if total_cost < best_cost:
                best_cost = total_cost
        else:
            for i in range(n):
                if i not in current.path:
                    cost_next = current.cost + mat[current.path[-1], i]
                    queue.put(Node(current.path + [i], cost_next, current.level + 1))
    print("Branch-and-Bound concluído.")
    return best_cost

def pre_order_dfs(graph, starting_node):
    # Controle dos vértices já visitados
    seen_vertices = set()

    traversal = []
    def dfs(node):
        if node not in seen_vertices:
            # Visitando primeiramente o nó pai (pois é um caminhamento em pré-ordem)
            seen_vertices.add(node)
            traversal.append(node)

            # Em sequência, visitando os filhos em ordem
            for neighbor in graph.neighbors(node):
                dfs(neighbor)
    
    dfs(starting_node)

    return traversal

@timeout(TIME_LIMIT)
def twice_around_tree(cities_distances):
    print("Executando Twice-Around-the-Tree...")

    cities_graph = nx.Graph()

    num_cities = len(cities_distances)
    # Adicionando as rotas entre as cidades ao grafo acima sem duplicá-las
    for i in range(num_cities):
        for j in range(i+1, num_cities):
            cities_graph.add_edge(i, j, weight=cities_distances[i, j])

    mst = nx.minimum_spanning_tree(cities_graph)
    
    #traversal = list(nx.dfs_preorder_nodes(mst, source=0))
    
    # Computando uma Depth-First Search em pré-ordem no grafo
    # É o equivalente a encontrar um circuito Euleriano na MST com as arestas duplicadas e posteriormente percorrer o mesmo removendo arestas repetidas
    traversal = pre_order_dfs(mst, mst.nodes[0])
    
    # Fechando o ciclo do TS e computando seu custo
    complete_ts_tour = traversal + [traversal[0]]

    tour_length = len(complete_ts_tour)
    tour_cost = sum(cities_distances[complete_ts_tour[i], complete_ts_tour[i+1]] for i in range(tour_length-1))
                      
    print("Twice-Around-the-Tree concluído.")

    return tour_cost

def compute_eulerian_circuit(graph, starting_node):
    nodes_visiting_stack = [starting_node]

    current_path = []
    while nodes_visiting_stack:
        # Consultando o nó no topo da lista
        current_node = nodes_visiting_stack[-1]
        neighbors = list(graph.neighbors(current_node))
        
        if neighbors:
            next_node = neighbors[0]
            graph.remove_edge(current_node, next_node)
            nodes_visiting_stack.append(next_node)
        else:
            # Voltando para o vértice pai, caso não tenhamos outra opção
            current_path.append(nodes_visiting_stack.pop())

    eulerian_circuit = []  # Stores the sequence of edges in the Eulerian circuit
    for i in range(len(current_path) - 1):
        eulerian_circuit.append((current_path[i], current_path[i + 1]))
    
    return eulerian_circuit

@timeout(TIME_LIMIT)
def christofides(cities_distances):
    print("Executando Christofides...")

    cities_graph = nx.Graph()

    num_cities = len(cities_distances)
    # Adicionando as rotas entre as cidades ao grafo acima sem duplicá-las
    for i in range(num_cities):
        for j in range(i+1, num_cities):
            cities_graph.add_edge(i, j, weight=cities_distances[i, j])

    mst = nx.minimum_spanning_tree(cities_graph)

    # Selecionando apenas os nós de grau ímpar da MST e computando o emparelhamento de custo mínimo no subgrafo induzido por eles
    odd_degree_nodes = [v for v, deg in mst.degree() if deg % 2 == 1]

    odd_degree_nodes_subgraph = cities_graph.subgraph(odd_degree_nodes)

    min_weight_matching = nx.algorithms.matching.min_weight_matching(odd_degree_nodes_subgraph, weight='weight')

    # Combinando o emparelhamento com a MST
    mst_min_match = nx.MultiGraph(mst)
    mst_min_match.add_edges_from(min_weight_matching)

    #euler_circuit = list(nx.eulerian_circuit(mst_min_match))
    euleurian_circuit = compute_eulerian_circuit(mst_min_match, mst_min_match.nodes[0])

    # Removendo vértices duplicados no caminho euleriano (válido pela desigualdade triangular)
    seen_vertices = set()
    
    traversal = []
    for u, _ in euleurian_circuit:
        if u not in seen_vertices:
            traversal.append(u)
            seen_vertices.add(u)

     # Fechando o ciclo do TS
    complete_ts_tour = traversal + [traversal[0]]
    
    tour_length = len(complete_ts_tour)
    tour_cost = sum(cities_distances[complete_ts_tour[i], complete_ts_tour[i+1]] for i in range(tour_length-1))

    print("Christofides concluído.")

    return tour_cost

def list_instances(directory):
    """Lista instâncias disponíveis no diretório."""
    if not os.path.isdir(directory):
        print(f"Diretório não encontrado: {directory}")
        return []
    return [
        os.path.splitext(f)[0]
        for f in os.listdir(directory)
        if f.endswith('.tsp')
    ]

def ratio_to_optimum(found_cost, optimum):
    """Calcula a razão entre custo encontrado e ótimo."""
    if found_cost == 'NA' or optimum == 'NA' or not isinstance(optimum, (int, float)) or optimum == 0:
        return 'NA'
    return round(found_cost / optimum, 4)

def run_experiments(dataset_dir, opt_solutions, output_file="results.csv"):
    """Executa experimentos e salva resultados.
    Utiliza gerenciamento de contexto para garantir fechamento adequado dos arquivos.
    """
    import contextlib
    
    @contextlib.contextmanager
    def managed_file(filename, mode="a"):
        f = open(filename, mode, buffering=1)  # Line buffering
        try:
            yield f
        finally:
            f.flush()
            f.close()

    # Escreve cabeçalho
    with managed_file(output_file, "w") as f:
        f.write("file,optimum,"
                "bnb_cost,bnb_time,bnb_mem,bnb_ratio,"  # tempo em segundos, mem em KB
                "tat_cost,tat_time,tat_mem,tat_ratio,"   
                "christ_cost,christ_time,christ_mem,christ_ratio\n")  

    for inst in list_instances(dataset_dir):
        print(f"\nProcessando instância: {inst}")
        coords = read_tsp_file(os.path.join(dataset_dir, inst + ".tsp"))
        if not coords:
            print(f"Erro ao processar {inst}: coordenadas vazias.")
            continue
        mat = create_distance_matrix(coords)
        optimum = opt_solutions.get(inst, 'NA')

        # Branch-and-Bound
        try:
            bnb_cost, bnb_time, bnb_mem = measure_memory_and_time(bnb_algorithm, mat)
            print(f"Branch-and-Bound finalizado: Custo = {bnb_cost}")
        except TimeoutError:
            bnb_cost, bnb_time, bnb_mem = 'NA', 'NA', 'NA'
            print("Branch-and-Bound atingiu o tempo limite.")
        bnb_ratio = ratio_to_optimum(bnb_cost, optimum)

        # TAT
        try:
            tat_cost, tat_time, tat_mem = measure_memory_and_time(twice_around_tree, mat)
            print(f"Twice-Around-the-Tree finalizado: Custo = {tat_cost}")
        except Exception as e:
            tat_cost, tat_time, tat_mem = 'NA', 'NA', 'NA'
            print(f"Erro em Twice-Around-the-Tree: {e}")
        tat_ratio = ratio_to_optimum(tat_cost, optimum)

        # Christofides
        try:
            christ_cost, christ_time, christ_mem = measure_memory_and_time(christofides, mat)
            print(f"Christofides finalizado: Custo = {christ_cost}")
        except Exception as e:
            christ_cost, christ_time, christ_mem = 'NA', 'NA', 'NA'
            print(f"Erro em Christofides: {e}")
        christ_ratio = ratio_to_optimum(christ_cost, optimum)

        # Escrita dos resultados
        with managed_file(output_file) as f:
            f.write(
                f"{inst},{optimum},"
                f"{bnb_cost},{bnb_time},{bnb_mem},{bnb_ratio},"
                f"{tat_cost},{tat_time},{tat_mem},{tat_ratio},"
                f"{christ_cost},{christ_time},{christ_mem},{christ_ratio}\n"
            )

if __name__ == "__main__":
    opt_file = "optimal_solutions.txt"
    optimal_solutions = load_optimal_solutions(opt_file)
    run_experiments("all_tsp", optimal_solutions)