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

@timeout(TIME_LIMIT)
def tat_algorithm(mat):
    """Twice-Around-the-Tree."""
    print("Executando Twice-Around-the-Tree...")
    n = len(mat)
    g = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            g.add_edge(i, j, weight=mat[i, j])
    mst = nx.minimum_spanning_tree(g)
    traversal = list(nx.dfs_preorder_nodes(mst, source=0))
    tour = traversal + [traversal[0]]
    print("Twice-Around-the-Tree concluído.")
    return sum(mat[tour[i], tour[i+1]] for i in range(len(tour)-1))

@timeout(TIME_LIMIT)
def christofides_algorithm(mat):
    """Christofides."""
    print("Executando Christofides...")
    n = len(mat)
    g = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            g.add_edge(i, j, weight=mat[i, j])
    mst = nx.minimum_spanning_tree(g)
    odd_nodes = [v for v, deg in mst.degree() if deg % 2 == 1]
    subg = g.subgraph(odd_nodes)
    matching = nx.algorithms.matching.min_weight_matching(subg, weight='weight')
    mst_aug = nx.MultiGraph(mst)
    mst_aug.add_edges_from(matching)
    euler_circuit = list(nx.eulerian_circuit(mst_aug))
    path = []
    visited = set()
    for u, _ in euler_circuit:
        if u not in visited:
            path.append(u)
            visited.add(u)
    path.append(path[0])
    print("Christofides concluído.")
    return sum(mat[path[i], path[i+1]] for i in range(len(path)-1))

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
            tat_cost, tat_time, tat_mem = measure_memory_and_time(tat_algorithm, mat)
            print(f"Twice-Around-the-Tree finalizado: Custo = {tat_cost}")
        except Exception as e:
            tat_cost, tat_time, tat_mem = 'NA', 'NA', 'NA'
            print(f"Erro em Twice-Around-the-Tree: {e}")
        tat_ratio = ratio_to_optimum(tat_cost, optimum)

        # Christofides
        try:
            christ_cost, christ_time, christ_mem = measure_memory_and_time(christofides_algorithm, mat)
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