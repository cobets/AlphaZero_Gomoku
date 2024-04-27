import numpy as np
from dotsenv import DotsEnv, BLACK


class DotsBoard:
    def __init__(self, width, height, n_in_row):
        self.width = width
        self.height = height
        self.players = [1, 2]
        self.dots_env = None
        self.availables = None
        self.last_move = None
        self.start_player = None
        self.current_player = None
        self.next_player = None
        self.init_board()

    def init_board(self, start_player=0):
        self.dots_env = DotsEnv(self.width, self.height)
        self.availables = self.dots_env.legal_actions()
        self.last_move = -1
        self.start_player = self.players[start_player]  # start player
        self.current_player = self.start_player
        self.next_player = 2 if self.start_player == 1 else 1

    def current_state(self):
        l_feature = self.dots_env.feature()
        l_last_move = np.zeros((self.width, self.height), dtype=np.float32)

        if self.last_move > -1:
            l_last_move[self.last_move // self.width, self.last_move % self.height] = 1.0

        if self.dots_env.player == BLACK:
            l_player = np.ones((self.width, self.height), dtype=np.float32)
        else:
            l_player = np.zeros((self.width, self.height), dtype=np.float32)

        l_result = np.stack([
            l_feature[0],
            l_feature[1],
            l_last_move,
            l_player
        ])

        return l_result

    def game_end(self):
        l_terminal = self.dots_env.terminal()
        l_winner = -1

        if l_terminal:
            l_rerminal_reward = self.dots_env.terminal_reward()
            if l_rerminal_reward > 0:
                l_winner = self.start_player
            elif l_rerminal_reward < 0:
                l_winner = self.next_player

        return l_terminal, l_winner

    def get_current_player(self):
        return self.start_player if self.dots_env.player == BLACK else self.next_player

    def do_move(self, move):
        self.dots_env.play(move)
        self.availables = self.dots_env.legal_actions()
        self.last_move = move
        self.current_player = self.get_current_player()

