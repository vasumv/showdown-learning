import numpy as np

class Converter(object):

    def __init__(self):

        self.poke_index = 0
        self.poke_forward_mapping = {}
        self.poke_backward_mapping = []

        self.move_index = 0
        self.move_forward_mapping = {}
        self.move_backward_mapping = []

    def learn_encodings(self, experiences):
        for experience in experiences:
            state, action, _, _ = experience
            for team in [0, 1]:
                for poke in state.get_team(team):
                    poke = self.convert_poke_name(poke.name)
                    if poke not in self.poke_forward_mapping:
                        self.poke_forward_mapping[poke] = self.poke_index
                        self.poke_backward_mapping.append(poke)
                        self.poke_index += 1
            if action.is_move():
                move = action.name
                if move not in self.move_forward_mapping:
                    self.move_forward_mapping[move] = self.move_index
                    self.move_backward_mapping.append(move)
                    self.move_index += 1

            if action.is_switch():
                poke = action.name
                poke = self.convert_poke_name(poke)
                if poke not in self.poke_forward_mapping:
                    self.poke_forward_mapping[poke] = self.poke_index
                    self.poke_backward_mapping.append(poke)
                    self.poke_index += 1

    def encode_poke(self, poke):
        poke = self.convert_poke_name(poke)
        return np.eye(self.poke_index)[self.poke_forward_mapping[poke]]

    def encode_move(self, move):
        return np.eye(self.move_index)[self.move_forward_mapping[move]]

    def convert_poke_name(self, poke):
        return poke.split(',')[0]

    def encode_state(self, state):
        poke = state.get_primary(0)
        poke = self.convert_poke_name(poke.name)
        my_team = [self.encode_poke(p.name) for p in state.get_team(0)]
        my_team_vec = [0] * self.poke_index
        their_team = [self.encode_poke(p.name) for p in state.get_team(0)]
        their_team_vec = [0] * self.poke_index
        for my_poke, their_poke in zip(my_team[1:], their_team[1:]):
            my_team_vec[my_poke.argmax()] = 1
            their_team_vec[their_poke.argmax()] = 1
        x = np.concatenate([
            self.encode_poke(state.get_primary(0).name),
            my_team_vec,
            state.get_faints(0) + [1] * (6 - len(my_team)),
            state.get_healths(0) + [0] * (6 - len(my_team)),
            self.encode_poke(state.get_primary(1).name),
            their_team_vec,
            state.get_faints(1) + [1] * (6 - len(their_team)),
            state.get_healths(1) + [0] * (6 - len(their_team)),
        ])
        return x.astype(np.float32)

    def encode_action(self, action):
        if action.is_move():
            y = self.encode_move(action.name).tolist() + [0] * self.poke_index
        else:
            switch_poke = self.convert_poke_name(action.name)
            y = [0] * self.move_index + self.encode_poke(switch_poke).tolist()
        return np.array(y).astype(np.float32)

    def get_input_dimension(self):
        return self.poke_index * 4 + 24

    def get_output_dimension(self):
        return self.move_index + self.poke_index
