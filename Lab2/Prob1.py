import heapq
from collections import namedtuple

State = namedtuple("State", ["f", "board", "zero_idx", "g", "h"])


def manhattan_distance(board):
    distance = 0
    for i in range(9):
        if board[i] != 0:
            correct_pos = (board[i] - 1) % 8
            distance += abs(i // 3 - correct_pos // 3) + abs(i % 3 - correct_pos % 3)
    return distance


def solve(initial_board):
    goal_board = (1, 3, 5, 7, 0, 2, 6, 8, 4)
    f = manhattan_distance(initial_board)
    initial_state = State(f, initial_board, initial_board.index(0), 0, manhattan_distance(initial_board))

    open_set = [initial_state]
    heapq.heapify(open_set)

    visited = set()

    while open_set:
        current_state = heapq.heappop(open_set)

        if current_state.board == goal_board:
            return current_state.g

        visited.add(current_state.board)

        x, y = current_state.zero_idx // 3, current_state.zero_idx % 3
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                new_zero_idx = new_x * 3 + new_y
                new_board = list(current_state.board)
                new_board[current_state.zero_idx], new_board[new_zero_idx] = new_board[new_zero_idx], new_board[current_state.zero_idx]
                new_board = tuple(new_board)
                if new_board not in visited:
                    f = current_state.g + 1 + manhattan_distance(new_board)
                    new_state = State(f, new_board, new_zero_idx, current_state.g + 1, manhattan_distance(new_board))
                    heapq.heappush(open_set, new_state)


if __name__ == "__main__":
    input_str = ["135720684", "105732684", "015732684", "135782604", "715032684"]
    for i in range(len(input_str)):
        initial_board = tuple(map(int, input_str[i]))
        print(solve(initial_board))