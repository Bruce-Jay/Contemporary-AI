import heapq
from collections import namedtuple

State = namedtuple("State", ["f", "room", "g", "h"])

def heuristic(room, adj_list):
    return min(adj_list[room], default=0)

def k_shortest_paths(N, K, adj_list):
    # print(len(adj_list))
    start_room = N
    goal_room = 1
    visited = set()
    paths = []

    f = heuristic(start_room, adj_list)[1]
    initial_state = State(f, start_room, 0, heuristic(start_room, adj_list))
    open_set = [initial_state]
    heapq.heapify(open_set)

    while open_set and len(paths) < K:
        current_state = heapq.heappop(open_set)

        if current_state.room == goal_room:
            paths.append(current_state.g)
            continue

        visited.add(current_state)

        for next_room, cost in adj_list[current_state.room]:
            if State(f, next_room, current_state.g + cost, heuristic(next_room, adj_list)) not in visited:
                if isinstance(heuristic(next_room, adj_list), tuple):
                    f = current_state.g + cost + heuristic(next_room, adj_list)[1]
                else:
                    f = current_state.g + cost
                new_state = State(f, next_room, current_state.g + cost, heuristic(next_room, adj_list))
                heapq.heappush(open_set, new_state)

    if len(paths) < K:
        paths.extend([-1] * (K-len(paths)))

    return paths

if __name__ == "__main__":
    input_strs = ["5 6 4\n1 2 1\n1 3 1\n2 4 2\n2 5 2\n3 4 2\n3 5 2",
                "6 9 4\n1 2 1\n1 3 3\n2 4 2\n2 5 3\n3 6 1\n4 6 3\n5 6 3\n1 6 8\n2 6 4",
                "7 12 6\n1 2 1\n1 3 3\n2 4 2\n2 5 3\n3 6 1\n4 7 3\n5 7 1\n6 7 2\n1 7 10\n2 6 4\n3 4 2\n4 5 1",
                "5 8 7\n1 2 1\n1 3 3\n2 4 1\n2 5 3\n3 4 2\n3 5 2\n1 4 3\n1 5 4",
                "6 10 8\n1 2 1\n1 3 2\n2 4 2\n2 5 3\n3 6 3\n4 6 3\n5 6 1\n1 6 8\n2 6 5\n3 4 1"]
    for input_str in input_strs:
        lines = input_str.strip().split("\n")
        N, M, K = map(int, lines[0].split())
        adj_list = {i: [] for i in range(1, N + 1)}

        for line in lines[1:]:
            x, y, d = map(int, line.split())
            if x < y:
                x, y = y, x
            adj_list[x].append((y, d))

        paths = k_shortest_paths(N, K, adj_list)
        for path in paths:
            print(path, end=" ")
        print("\n")
