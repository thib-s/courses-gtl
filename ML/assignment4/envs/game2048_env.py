from __future__ import print_function

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

import itertools
import logging
from six import StringIO
import sys


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class IllegalMove(Exception):
    pass


class Game2048Env(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, size=4, computeP=False):
        # Definitions for game. Board must be square.
        self.size = size
        self.w = self.size
        self.h = self.size
        squares = self.size * self.size
        self.nS = 0
        self.nA = 4
        if computeP:
            assert size == 2
            self.P = [[[(1, 0, 0, True)]] * 4] * 1554
            self.nS = 1554
            state = np.zeros((self.h, self.w), np.int)
            for v11 in range(6):
                for v12 in range(6):
                    for v21 in range(6):
                        for v22 in range(6):
                            state[0, 0] = 2**v11 if v11!=0 else 0
                            state[0, 1] = 2**v12 if v12!=0 else 0
                            state[1, 0] = 2**v21 if v21!=0 else 0
                            state[1, 1] = 2**v22 if v22!=0 else 0
                            action = []
                            for a in range(4):
                                try:
                                    move_score, board = self.move_test(np.copy(state), a)
                                    empties = self.empties_test(board)
                                    to_append = []
                                    for (x, y) in empties:
                                        prob = (0.8/len(empties))
                                        next_state = np.copy(board)
                                        next_state[x, y] = 2
                                        to_append.append((prob, self.board_to_id(next_state), move_score, False))
                                        prob = (0.2 / len(empties))
                                        next_state = np.copy(board)
                                        next_state[x, y] = 4
                                        to_append.append((prob, self.board_to_id(next_state), move_score, False))
                                except IllegalMove:
                                    to_append = [(1, self.board_to_id(state), 0, True)]
                                action.append(to_append)
                            if (np.sum(state == 32) <= 1) and (np.sum(state == 16) <= 1):
                                self.P[self.board_to_id(state)] = action

        # Maintain own idea of game score, separate from rewards
        self.score = 0

        # Members for gym implementation
        self.action_space = spaces.Discrete(4)
        # Suppose that the maximum tile is as if you have powers of 2 across the board.
        self.observation_space = spaces.Box(0, 2 ** squares, (self.w * self.h,))
        # Guess that the maximum reward is also 2**squares though you'll probably never get that.
        self.reward_range = (0., float(2 ** squares))

        # Initialise seed
        self._seed()

        # Reset ready for a game
        self._reset()

    def board_to_id(self, board):
        return int(np.log2(board[0, 0] if board[0, 0] != 0 else 1)) +\
               6*int(np.log2(board[0, 1] if board[0, 1] != 0 else 1))+ \
               6*6*int(np.log2(board[1, 0] if board[1, 0] != 0 else 1)) +\
               6*6*6*int(np.log2(board[1, 1] if board[1, 1] != 0 else 1))

    def seed(self, seed=None):
        return self._seed(seed)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        (obs, reward, done, info) = self._step(action)
        return self.board_to_id(self.Matrix), reward, done, info

    # Implement gym interface
    def _step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile."""
        logging.debug("Action {}".format(action))
        score = 0
        done = None
        try:
            score = float(self.move(action))
            self.score += score
            # assert score <= 2 ** (self.w * self.h)
            self.add_tile()
            done = self.isend()
            reward = float(score)
        except IllegalMove as e:
            logging.debug("Illegal move")
            done = False
            # No reward for illegal move
            reward = 0.

        # print("Am I done? {}".format(done))
        observation = self.Matrix.flatten()
        info = dict()
        return observation, reward, done, info
        # Return observation (board state), reward, done and info dict

    def reset(self):
        self._reset()
        return 0

    def _reset(self):
        self.Matrix = np.zeros((self.h, self.w), np.int)
        self.score = 0

        logging.debug("Adding tiles")
        self.add_tile()
        self.add_tile()

        return self.Matrix.flatten()

    def render(self, mode='human'):
        return self._render(mode)

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = 'Score: {}\n'.format(self.score)
        s += 'Highest: {}\n'.format(self.highest())
        npa = np.array(self.Matrix)
        grid = npa.reshape((self.size, self.size))
        s += "{}\n".format(grid)
        outfile.write(s)
        return outfile

    # Implement 2048 game
    def add_tile(self):
        """Add a tile, probably a 2 but maybe a 4"""
        val = 0
        if self.np_random.random_sample() > 0.8:
            val = 4
        else:
            val = 2
        empties = self.empties()
        assert empties
        empty_idx = self.np_random.choice(len(empties))
        empty = empties[empty_idx]
        logging.debug("Adding %s at %s", val, (empty[0], empty[1]))
        self.set(empty[0], empty[1], val)

    def get(self, x, y):
        """Return the value of one square."""
        return self.Matrix[x, y]

    def set(self, x, y, val):
        """Set the value of one square."""
        self.Matrix[x, y] = val

    def empties(self):
        """Return a list of tuples of the location of empty squares."""
        empties = list()
        for y in range(self.h):
            for x in range(self.w):
                if self.get(x, y) == 0:
                    empties.append((x, y))
        return empties

    def empties_test(self, board):
        """Return a list of tuples of the location of empty squares."""
        empties = list()
        for y in range(self.h):
            for x in range(self.w):
                if board[x, y] == 0:
                    empties.append((x, y))
        return empties

    def highest(self):
        """Report the highest tile on the board."""
        highest = 0
        for y in range(self.h):
            for x in range(self.w):
                highest = max(highest, self.get(x, y))
        return highest

    def move_test(self, board, direction):
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the score that [would have] got."""
        board = np.copy(board)

        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two  # 0 for towards up left, 1 for towards bottom right

        # Construct a range for extracting row/column into a list
        rx = list(range(2))
        ry = list(range(2))

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(0):
                old = [board[x, y] for x in rx]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    for x in rx:
                        board[x, y] = new[x]
        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = [board[x, y] for y in ry]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    for y in ry:
                        board[x, y] = new[y]
        if changed != True:
            raise IllegalMove

        return move_score, board

    def move(self, direction, trial=False):
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the score that [would have] got."""
        if not trial:
            if direction == 0:
                logging.debug("Up")
            elif direction == 1:
                logging.debug("Right")
            elif direction == 2:
                logging.debug("Down")
            elif direction == 3:
                logging.debug("Left")

        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two  # 0 for towards up left, 1 for towards bottom right

        # Construct a range for extracting row/column into a list
        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(self.h):
                old = [self.get(x, y) for x in rx]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.set(x, y, new[x])
        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = [self.get(x, y) for y in ry]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.set(x, y, new[y])
        if changed != True:
            raise IllegalMove

        return move_score

    def combine(self, shifted_row):
        """Combine same tiles when moving to one side. This function always
           shifts towards the left. Also count the score of combined tiles."""
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):
            if skip:
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_score += p[0] + p[1]
                # Skip the next thing in the list.
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return (combined_row, move_score)

    def shift(self, row, direction):
        """Shift one row left (direction == 0) or right (direction == 1), combining if required."""
        length = len(row)
        assert length == self.size
        assert direction == 0 or direction == 1

        # Shift all non-zero digits up
        shifted_row = [i for i in row if i != 0]

        # Reverse list to handle shifting to the right
        if direction:
            shifted_row.reverse()

        (combined_row, move_score) = self.combine(shifted_row)

        # Reverse list to handle shifting to the right
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)

    def isend(self):
        """Has the game ended. Game ends if there is a 2048 tile or there are
        no legal moves. If there are empty spaces then there must be legal
        moves."""

        if self.highest() == 2048:
            return True

        for direction in range(4):
            try:
                self.move(direction, trial=True)
                # Not the end if we can do any move
                return False
            except IllegalMove:
                pass
        return True

    def get_board(self):
        """Retrieve the whole board, useful for testing."""
        return self.Matrix

    def set_board(self, new_board):
        """Retrieve the whole board, useful for testing."""
        self.Matrix = new_board
