import os
import time
import math
import tracemalloc
import numpy as np
import networkx as nx
from queue import PriorityQueue
from timeout_decorator import timeout, TimeoutError

TIME_LIMIT = 1800  # Tempo limite de execução (segundos)

def read_tsp_file(path):
    """
    Lê as coordenadas dos nós a partir de um arquivo TSPLIB.
    
    Parâmetros:
    path: Caminho para o arquivo `.tsp` que contém as coordenadas dos nós.
    
    Retorno:
    coords: Lista contendo as coordenadas dos nós como tuplas de floats (x, y).
          Retorna uma lista vazia se o arquivo não for encontrado ou ocorrer um erro.
    """
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
    """
    Carrega as soluções ótimas de um arquivo, associando o nome da instância ao seu custo ótimo.
    
    Parâmetros:
    path: Caminho para o arquivo que contém as soluções ótimas.
    
    Retorno:
    sols: Dicionário onde as chaves são os nomes das instâncias e os valores são os custos ótimos correspondentes.
          Retorna um dicionário vazio se o arquivo não for encontrado ou ocorrer um erro.
    """
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
    """
    Gera uma matriz de distâncias euclidianas entre os pontos dados.
    
    Parâmetros:
    coords: Lista de tuplas (x, y) representando as coordenadas dos pontos.
    
    Retorno:
    mat: Matriz de distâncias entre os pontos, onde o valor na posição [i][j] é a distância 
         euclidiana entre o ponto i e o ponto j.
    """
    n = len(coords)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            mat[i][j] = dist
            mat[j][i] = dist

    return mat

def measure_memory_and_time(func, *args, **kwargs):
    """
    Mede o tempo de execução e o pico de memória utilizado durante a execução de uma função.
    
    Parâmetros:
    func: Função a ser executada e medida.
    *args, **kwargs: Argumentos e parâmetros a serem passados para a função.
    
    Retorno:
    tuple: Tupla contendo três valores:
            - result: Resultado da execução da função.
            - elapsed_time: Tempo de execução em segundos.
            - peak_mem: Pico de memória utilizado em KB.    
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

class Node:
    """
    Representa um nó no algoritmo Branch-and-Bound para resolver o problema do Caixeiro Viajante (TSP).
    
    Atributos:
    current_path: Lista de vértices representando o caminho percorrido até o momento.
    level: Nível no qual o nó se encontra (número de vértices no caminho).
    current_path_cost: Custo total do caminho até o momento.
    bound: Estimativa inferior para o custo do caminho completo a partir deste nó.
    
    Métodos:
    __lt__(self, other): Compara dois nós com base no bound (utilizado para priorizar nós na fila de prioridades).
    """
    def __init__(self, current_path, level, cities_distances):
        self.current_path = current_path
        self.level = level
        self.current_path_cost = 0 
        if len(current_path) > 1: 
            self.current_path_cost = sum([cities_distances[current_path[i], current_path[i+1]] for i in range(len(current_path)-1)])
        self.bound = compute_bound(cities_distances, current_path)

    def __lt__(self, other):
        return self.bound < other.bound

def compute_bound(cities_distances, current_path):
    """
    Calcula uma estimativa inferior para o custo de um caminho baseado na matriz de distâncias.
    
    Parâmetros:
    cities_distances: Matriz de distâncias entre os vértices.
    current_path: Caminho até o momento representado por uma lista de índices de vértices.
    
    Retorno:
    bound: Estimativa inferior para o custo do caminho completo, considerando os vértices já visitados e as menores distâncias 
           não visitadas.
    """
    bound = 0
    used_edges = set()
    remaining_edges_per_vertex = np.full(len(cities_distances), 2)
    
    for i in range(len(current_path)-1):
        remaining_edges_per_vertex[current_path[i]] -= 1
        remaining_edges_per_vertex[current_path[i+1]] -= 1

        used_edges.add((current_path[i], current_path[i+1]))
        used_edges.add((current_path[i+1], current_path[i]))

        bound += cities_distances[current_path[i], current_path[i+1]]

    num_cities = len(cities_distances)
    for i in range(num_cities):
        current_city_non_visited_paths_costs = []

        for j in range(num_cities):
            if (i != j) and ((i, j) not in used_edges):
                current_city_non_visited_paths_costs.append(cities_distances[i, j])

        best_non_visited_paths = sorted(np.array(current_city_non_visited_paths_costs))
        best_non_visited_paths = best_non_visited_paths[0:remaining_edges_per_vertex[i]]

        for cost in best_non_visited_paths:
            bound += cost

    return math.ceil(bound/2)

@timeout(TIME_LIMIT)
def branch_and_bound(cities_distances):
    """
    Função que computa um algoritmo Branch-and-Bound para o TSP.
    
    Parâmetros:
    cities_distances: Matriz de distâncias entre os vértices.
    
    Retorno:
    best_cost: Custo do caminho mais curto encontrado.
    """
    cities_distances = np.matrix(cities_distances)

    best_cost = float('inf')
    num_cities = len(cities_distances)

    # Como exploraremos primeiro o nó de menor bound, empregaremos uma fila de prioridades    
    queue = PriorityQueue()

    # Sem perda de generalidade, podemos considerar que o tour começa no vértice arbitrário de índice 0
    queue.put(Node(np.array([0]), 0, cities_distances))

    while not queue.empty():
        # Explorando o nó mais promissor de acordo com a estratégia Best-First
        current_node = queue.get()

        #print(f"Explorando o nó {current_node.current_path}!")

        # Podando a árvore caso o bound é não menor do que o melhor caminho encontrado até então
        if current_node.bound >= best_cost:
            #print("Esse só não é promissor e será podado!")
            continue
        # Se chegamos no nível n-1, já que começamos do nível 0, podemos apenas retornar para o vértice inicial do caminho
        elif current_node.level == num_cities - 1:
            #print("Chegamos no último nível - iremos computar o custo do TSP!")
            total_cost = current_node.current_path_cost + cities_distances[current_node.current_path[-1], current_node.current_path[0]]
            if total_cost < best_cost:
                #print(f"Esse é o melhor tour até agora, com custo {total_cost}!")
                best_cost = total_cost
            else:
                pass
                #print(f"Esse tour não é mais barato do que já temos, já que ele tem custo {total_cost}!")
        # Visitando todos os nós representando vértices que ainda não foram visitados no caminho atual
        # Colocando cada nó na fila de prioridades de acordo com o seu bound estimado
        else:
            for i in range(num_cities):
                if i not in current_node.current_path:
                    new_path = np.append(current_node.current_path, i)
                    queue.put(Node(new_path, current_node.level + 1, cities_distances))
                    #print(f"Colocando {new_path} na fila!")

    #print(f"Encontramos o caminho mais curto de custo {best_cost}!")

    return best_cost

def pre_order_dfs(graph, starting_node):    
    """
    Função que computa uma Depth-First Search em pré-ordem em um grafo de entrada. 
    
    Parâmetros:
    graph: Grafo alvo.
    starting_node: Vértice em que iremos começar busca.
    
    Retorno:
    traversal: Lista contendo os vértices na ordem em que foram visitados.
    """
    # Controle dos vértices já visitados
    seen_vertices = set()

    traversal = []
    def aux_pre_order_dfs(node):
        if node not in seen_vertices:
            # Visitando primeiramente o nó pai (pois é um caminhamento em pré-ordem)
            seen_vertices.add(node)
            traversal.append(node)
            
            # Em sequência, visitando os filhos em ordem
            for neighbor in graph.neighbors(node):
                aux_pre_order_dfs(neighbor)

    aux_pre_order_dfs(starting_node)

    return traversal

@timeout(TIME_LIMIT)
def twice_around_tree(cities_distances):
    """
    Função que computa o algoritmo aproximativo Twice-Around-the-Tree para o TSP.

    Parâmetros:
    cities_distances: Matriz de adjacência que represente a instância de interesse do TSP.

    Retorno:
    tour_cost: Custo do tour encontrado (que é até 2x pior do que o ótimo real).
    """
    print("Executando Twice-Around-the-Tree...")

    cities_graph = nx.Graph()
    num_cities = len(cities_distances)

    # Adicionando as rotas entre as cidades ao grafo acima sem duplicá-las
    for i in range(num_cities):
        for j in range(i+1, num_cities):
            cities_graph.add_edge(i, j, weight=cities_distances[i, j])

    mst = nx.minimum_spanning_tree(cities_graph)

    # Percorrendo um circuito euleriando na MST com arestas duplicadas, removendo vértices duplicados
    starting_node = next(iter(mst.nodes))
    traversal = pre_order_dfs(mst, starting_node)
    
    # Fechando o ciclo do TS e computando seu custo
    complete_ts_tour = traversal + [traversal[0]]

    tour_length = len(complete_ts_tour)
    tour_cost = sum(cities_distances[complete_ts_tour[i], complete_ts_tour[i+1]] for i in range(tour_length-1))
                      
    print("Twice-Around-the-Tree concluído!")

    return tour_cost

def compute_eulerian_circuit(graph, starting_node):
    """
    Função que gera um circuito euleriano em um grafo de entrada. 
    
    Parâmetros:
    graph: Grafo alvo.
    starting_node: Vértice em que iremos começar o circuito.
    
    Retorno:
    traversal: Lista contendo uma sequência de vértices que formam um circuito euleriano.
    """
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

    eulerian_circuit = []
    for i in range(len(current_path) - 1):
        eulerian_circuit.append((current_path[i], current_path[i + 1]))
    
    return eulerian_circuit

@timeout(TIME_LIMIT)
def christofides(cities_distances):
    """
    Função que computa o algoritmo aproximativo de Christofides para o TSP.

    Parâmetros:
    cities_distances: Matriz de adjacência que represente a instância de interesse do TSP.

    Retorno:
    tour_cost: Custo do tour encontrado (que é até 1.5x pior do que o ótimo real).
    """
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


    # Computando um caminho euleriano no grafo resultante e removendo vértices repetidos (válido pela desigualdade triangular)
    starting_node = next(iter(mst_min_match.nodes))
    euleurian_circuit = compute_eulerian_circuit(mst_min_match, starting_node)

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
    """
    Lista as instâncias disponíveis no diretório, procurando por arquivos com a extensão '.tsp'.
    Ordena os arquivos pela DIMENSION especificada no conteúdo de cada arquivo.

    Parâmetros:
    directory: Caminho para o diretório onde as instâncias estão armazenadas.

    Retorno:
    list: Lista contendo os nomes das instâncias (sem a extensão '.tsp') ordenadas pelo valor de DIMENSION.
          Caso o diretório não seja encontrado, imprime uma mensagem de erro e retorna uma lista vazia.
    """
    if not os.path.isdir(directory):
        print(f"Diretório não encontrado: {directory}")
        return []

    instances = []

    for f in os.listdir(directory):
        if f.endswith('.tsp') and f != '.DS_Store':
            file_path = os.path.join(directory, f)
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if line.startswith("DIMENSION"):
                            dimension = int(line.split(':')[1].strip())
                            instances.append((os.path.splitext(f)[0], dimension))
                            break
            except Exception as e:
                print(f"Erro ao processar o arquivo {f}: {e}")

    # Sort instances by dimension
    sorted_instances = sorted(instances, key=lambda x: x[1])
    return [instance[0] for instance in sorted_instances]

def ratio_to_optimum(found_cost, optimum):
    """
    Calcula a razão entre o custo encontrado e o custo ótimo, com precisão de 4 casas decimais.
    
    Parâmetros:
    found_cost: Custo encontrado pela heurística ou algoritmo.
    optimum: Custo ótimo conhecido para a instância do problema.
    
    Retorno:
    float ou str: A razão entre o custo encontrado e o ótimo, arredondada para 4 casas decimais.
                  Se algum dos valores for inválido (ex: 'NA', 0 ou valores não numéricos), retorna 'NA'.
    """
    if found_cost == 'NA' or optimum == 'NA' or not isinstance(optimum, (int, float)) or optimum == 0:
        return 'NA'
    return round(found_cost / optimum, 4)

'''
def run_experiments(dataset_dir, opt_solutions, output_file="results.csv"):
    """
    Executa experimentos para várias instâncias, utilizando diferentes algoritmos e salva os resultados em um arquivo CSV.
    
    Parâmetros:
    dataset_dir: Diretório onde as instâncias do problema estão armazenadas.
    opt_solutions: Dicionário com as soluções ótimas para as instâncias.
    output_file: Nome do arquivo CSV onde os resultados serão salvos (padrão: "results.csv").
    
    Retorno:
    None: A função salva os resultados no arquivo especificado e não retorna nenhum valor.
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
            bnb_cost, bnb_time, bnb_mem = measure_memory_and_time(branch_and_bound, mat)
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
'''

def run_experiments(dataset_dir, opt_solutions, output_file_prefix="results", algorithm="all"):
    """
    Executa experimentos para várias instâncias usando algoritmos específicos e salva os resultados em arquivos CSV.
    
    Parâmetros:
    dataset_dir: Diretório onde as instâncias do problema estão armazenadas.
    opt_solutions: Dicionário com as soluções ótimas para as instâncias.
    output_file_prefix: Prefixo do nome do arquivo CSV onde os resultados serão salvos (padrão: "results").
    algorithm: Especifica o algoritmo a ser executado ('bnb', 'tat', 'christ', ou 'all' para executar todos).
    
    Retorno:
    None: A função salva os resultados no arquivo especificado e não retorna nenhum valor.
    """
    import contextlib

    def get_output_filename(algorithm):
        return f"{output_file_prefix}_{algorithm}.csv"

    @contextlib.contextmanager
    def managed_file(filename, mode="a"):
        f = open(filename, mode, buffering=1)  # Line buffering
        try:
            yield f
        finally:
            f.flush()
            f.close()

    # Header writing for each specified algorithm
    if algorithm in ['bnb', 'tat', 'christ']:
        with managed_file(get_output_filename(algorithm), "w") as f:
            f.write("file,optimum,cost,time,mem,ratio\n")
    elif algorithm == 'all':
        with managed_file(get_output_filename("all"), "w") as f:
            f.write("file,optimum,"
                    "bnb_cost,bnb_time,bnb_mem,bnb_ratio,"
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

        def run_and_record(func, name):
            try:
                cost, time, mem = measure_memory_and_time(func, mat)
                print(f"{name} finalizado: Custo = {cost}")
            except Exception as e:
                cost, time, mem = 'NA', 'NA', 'NA'
                print(f"Erro em {name}: {e}")
            ratio = ratio_to_optimum(cost, optimum)
            return cost, time, mem, ratio

        results = {}

        if algorithm in ['all', 'bnb']:
            bnb_results = run_and_record(branch_and_bound, "Branch-and-Bound")
            if algorithm == 'bnb':
                with managed_file(get_output_filename("bnb")) as f:
                    f.write(f"{inst},{optimum},{','.join(map(str, bnb_results))}\n")
            else:
                results.update({f"bnb_{k}": v for k, v in zip(["cost", "time", "mem", "ratio"], bnb_results)})

        if algorithm in ['all', 'tat']:
            tat_results = run_and_record(twice_around_tree, "Twice-Around-the-Tree")
            if algorithm == 'tat':
                with managed_file(get_output_filename("tat")) as f:
                    f.write(f"{inst},{optimum},{','.join(map(str, tat_results))}\n")
            else:
                results.update({f"tat_{k}": v for k, v in zip(["cost", "time", "mem", "ratio"], tat_results)})

        if algorithm in ['all', 'christ']:
            christ_results = run_and_record(christofides, "Christofides")
            if algorithm == 'christ':
                with managed_file(get_output_filename("christ")) as f:
                    f.write(f"{inst},{optimum},{','.join(map(str, christ_results))}\n")
            else:
                results.update({f"christ_{k}": v for k, v in zip(["cost", "time", "mem", "ratio"], christ_results)})

        if algorithm == 'all':
            with managed_file(get_output_filename("all")) as f:
                f.write(f"{inst},{optimum},"
                        f"{results['bnb_cost']},{results['bnb_time']},{results['bnb_mem']},{results['bnb_ratio']},"
                        f"{results['tat_cost']},{results['tat_time']},{results['tat_mem']},{results['tat_ratio']},"
                        f"{results['christ_cost']},{results['christ_time']},{results['christ_mem']},{results['christ_ratio']}\n")

def filter_2d_euc_tsp_instances(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                    for line in lines:
                        if line.startswith("EDGE_WEIGHT_TYPE"):
                            if "EUC_2D" not in line:
                                os.remove(file_path)
                                print(f"Deleted file: {file_path}")
                            break
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":

    # Carrega as soluções ótimas
    opt_file = "optimal_solutions.txt"
    optimal_solutions = load_optimal_solutions(opt_file)

    # Testes em instâncias pequenas
    # filter_2d_euc_tsp_instances('./small')     # Remove as instâncias cujas distâncias não são Euclidianas 2D
    # run_experiments("small", optimal_solutions, 'results_small', 'tat')
    # run_experiments("small", optimal_solutions, 'results_small', 'christ')
    # run_experiments("small", optimal_solutions, 'results_small', 'bnb')

    # Testes em todas as instâncias
    filter_2d_euc_tsp_instances('./all_tsp')     # Remove as instâncias cujas distâncias não são Euclidianas 2D
    # run_experiments("all_tsp", optimal_solutions, 'results', 'tat')
    run_experiments("all_tsp", optimal_solutions, 'results', 'christ')
    run_experiments("all_tsp", optimal_solutions, 'results', 'bnb')
