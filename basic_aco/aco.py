#!usr/bin/bash
""" Use ACO to solve Travelling Salesman Problem (TSP) """


class ACO(object):
    def __init__(self):
        pass


class Vertex(object):
    def __init__(self, _id):
        self.id = _id
        self.name = "City " + str(_id)
        self.adjacent_vertices = []

    def add_adjacent_vertex(self, vertex, weight):
        if isinstance(vertex, Vertex):
            edge = (vertex, weight)
            self.adjacent_vertices.append(edge)


def main():
    """ This graph is constructed to be the same as the graph found in the Ant Algorithms part 1.pdf """
    graph = []
    for i in range(9):
        graph.append(Vertex(i))
    graph[0].add_adjacent_vertex(graph[1], 4)
    graph[0].add_adjacent_vertex(graph[7], 8)

    graph[1].add_adjacent_vertex(graph[0], 4)
    graph[1].add_adjacent_vertex(graph[1], 8)
    graph[1].add_adjacent_vertex(graph[7], 11)

    graph[2].add_adjacent_vertex(graph[1], 8)
    graph[2].add_adjacent_vertex(graph[3], 7)
    graph[2].add_adjacent_vertex(graph[5], 4)
    graph[2].add_adjacent_vertex(graph[8], 2)

    graph[3].add_adjacent_vertex(graph[2], 7)
    graph[3].add_adjacent_vertex(graph[4], 9)
    graph[3].add_adjacent_vertex(graph[5], 14)

    graph[4].add_adjacent_vertex(graph[3], 9)
    graph[4].add_adjacent_vertex(graph[5], 10)

    graph[5].add_adjacent_vertex(graph[2], 4)
    graph[5].add_adjacent_vertex(graph[3], 14)
    graph[5].add_adjacent_vertex(graph[4], 10)
    graph[5].add_adjacent_vertex(graph[6], 2)

    graph[6].add_adjacent_vertex(graph[5], 2)
    graph[6].add_adjacent_vertex(graph[7], 1)
    graph[6].add_adjacent_vertex(graph[8], 6)

    graph[7].add_adjacent_vertex(graph[0], 8)
    graph[7].add_adjacent_vertex(graph[1], 11)
    graph[7].add_adjacent_vertex(graph[6], 1)
    graph[7].add_adjacent_vertex(graph[8], 7)

    graph[8].add_adjacent_vertex(graph[2], 2)
    graph[8].add_adjacent_vertex(graph[6], 6)
    graph[8].add_adjacent_vertex(graph[7], 7)


if __name__ == '__main__':
    main()
