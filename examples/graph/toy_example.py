from hash_graph import HashGraph, BFS

if __name__ == '__main__':
    from visualize import plot_graph
    import matplotlib.pyplot as plt

    '''
    Toy example on abstract graphs
    '''
    E = [
        ('A', 'B'),
        ('B', 'C'),
        ('C', 'D'),
        ('A', 'E'),
        ('E', 'D'),
    ]

    # Build a graph from transitions (edges)
    g = HashGraph()
    for t in range(5):
        for i, e in enumerate(E):
            g.insert(e[0], e[1], i * t)
    # plot_graph(g, with_labels=True)
    # plt.show()

    # Perform BFS over the graph, find the shortest path from D to others
    searcher = BFS()
    shortest_paths = (searcher.search(g, 'D', True))
    print('Shortest paths to D:', shortest_paths)

    # Retrieve data from the graph by the shortest path in a reverse order
    # We should see the numbers we insert when creating the graph
    for path_src, path in shortest_paths.items():
        for src, dst in reversed(path):
            print(f'{src}->{dst}:', g.edge(src, dst))
