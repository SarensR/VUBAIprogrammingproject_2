# FIRSTTRY.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random

import util
from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint


# Caetano Team
#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='DefensiveAgent', second='OffensiveAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class DefensiveAgent(CaptureAgent):
    """
  class holding choose_action and methods to be used bij def and off
  """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.opponent_start = None

        # initialized values
        self.minimax_depth = 2
        self.status = 'patrolling_up'
        self.patrolling_distance = 2
        self.patrol_route = []

        self.border = 0
        self.height = 0
        self.opponents = []
        self.observable_opponents = []
        self.distances_opponents = []

    def register_initial_state(self, game_state):
        """
        registers initial state
        """
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        # MAZE INFO
        width = game_state.data.layout.width - 2
        self.height = game_state.data.layout.height - 2
        self.border = int(width / 2)

        if self.red:
            self.opponent_start = (width, self.height)
        else:
            self.opponent_start = (1, 1)

        # DATA FILL IN
        self.opponents = self.get_opponents(game_state)

        # CALCULATIONS

        def find_patrol_points():
            if self.red:
                x_range = range(self.border, 1, -1)
            else:
                x_range = range(self.border, width)

            small_diff_bottom = 999
            small_diff_top = 999
            bottom_of_patrol = (self.border, 3)
            top_of_patrol = (self.border, self.height - 2)
            for x in x_range:
                if not game_state.has_wall(x, 3) and abs(
                        self.patrolling_distance - abs(x - self.border)) < small_diff_bottom:
                    small_diff_bottom = self.patrolling_distance - abs(x - self.border)
                    bottom_of_patrol = (x, 3)
                if not game_state.has_wall(x, self.height - 2) and abs(
                        self.patrolling_distance - abs(x - self.border)) < small_diff_top:
                    small_diff_top = self.patrolling_distance - abs(x - self.border)
                    top_of_patrol = (x, self.height - 2)
            return bottom_of_patrol, top_of_patrol

        self.patrol_route = find_patrol_points()

    def choose_action(self, game_state):
        """
        chase or patrol
        """
        # data fill in
        agent_state = game_state.get_agent_state(self.index)
        curr_pos = agent_state.get_position()
        self.update_observable_opponents_and_distances(game_state)
        if self.status == 'patrolling_down' and agent_state.get_position() == self.patrol_route[0]:
            self.status = 'patrolling_up'
        elif self.status == 'patrolling_up' and agent_state.get_position() == self.patrol_route[1]:
            self.status = 'patrolling_down'

        # chase or patrol
        if ((self.observable_opponents[0] and util.manhattanDistance(curr_pos, self.observable_opponents[0]) <= 5)
                or (self.observable_opponents[1] and util.manhattanDistance(curr_pos,
                                                                            self.observable_opponents[1]) <= 5)):
            return self.alpha_beta_minimax(game_state, self.minimax_depth, self.evaluation_function_defence)
        else:
            actions = game_state.get_legal_actions(self.index)
            if self.status == 'patrolling_down':
                patrol_point = self.patrol_route[0]
            else:
                patrol_point = self.patrol_route[1]
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(patrol_point, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def is_eaten(self, agent_index, game_state):
        if agent_index in self.opponents:
            return self.opponent_start == game_state.get_agent_position(agent_index)
        else:
            return self.start == game_state.get_agent_position(agent_index)

    def on_own_side(self, pos):
        """
        returns boolean indicating whether the position is on the agents own side.
        """
        if self.red:
            return pos[0] <= self.border
        else:
            return pos[0] > self.border

    def update_observable_opponents_and_distances(self, game_state):
        distances = game_state.get_agent_distances()
        self.observable_opponents = [False for i in range(game_state.get_num_agents())]
        for opponent in self.opponents:
            pos = game_state.get_agent_position(opponent)
            if pos:
                self.observable_opponents[opponent] = pos
                distances[opponent] = self.get_maze_distance(game_state.get_agent_position(self.index), pos)
            elif not pos and distances[opponent] <= 5:
                distances[opponent] = 6
        self.observable_opponents = [self.observable_opponents[index] for index in self.opponents]
        self.distances_opponents = [distances[index] for index in self.opponents]

    def evaluation_function_defence(self, game_state):
        """
        defensive evaluation function, expanded on DefensiveReflexAgent
        """
        features = util.Counter()
        agent_state = game_state.get_agent_state(self.index)
        pos = agent_state.get_position()
        self.update_observable_opponents_and_distances(game_state)

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if agent_state.is_pacman:
            features['on_defense'] = 0

        # Computes distance to invaders we can see
        invaders = [observable_opponent for observable_opponent in self.observable_opponents if observable_opponent
                    and self.on_own_side(observable_opponent)]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            features['invader_distance'] = min(self.distances_opponents)

        # what to do when scared_timer is active?
        features['scared_timer'] = agent_state.scared_timer

        # factor in the score
        features['score'] = game_state.get_score()

        # defended food left
        own_food_list = self.get_food_you_are_defending(game_state).as_list()
        features['own_food_left'] = len(own_food_list)

        features['is_eaten'] = self.is_eaten(self.index, game_state)

        weights = {'on_defense': 100, 'num_invaders': -10000, 'invader_distance': -30, 'scared_timer': -2,
                   'score': 2, 'own_food_left': 2, 'dis_opponent_to_its_side': -1, 'is_eaten': -1}

        return features * weights

    def alpha_beta_minimax(self, game_state, depth, evaluation):
        caller = self.index
        opponent1 = False
        opponent2 = False
        old_minimax_depth = self.minimax_depth
        max_number_recursions = pow(4, 2 * self.minimax_depth) + 2
        num_rec = util.Counter()
        num_rec["recs"] = 0

        opponents = []
        for ind, pos in enumerate(self.observable_opponents):
            if pos and ind == 0:
                opponents.append(self.opponents[0])
            elif pos and ind == 1:
                opponents.append(self.opponents[1])

        if len(opponents) == 0:
            return util.raiseNotDefined()
        elif len(opponents) == 1:
            opponent1 = opponents[0]
        else:
            self.minimax_depth = 1
            if caller == 0 or caller == 3:
                opponent1, opponent2 = self.opponents
            else:
                opponent2, opponent1 = self.opponents

        def mini_max(game_state, agent, depth_minimax, a=-float('inf'), b=float('inf')):
            num_rec["recs"] += 1
            if num_rec["recs"] >= max_number_recursions:
                return ['action?', evaluation(game_state)]
            if (depth_minimax == 0
                    or self.is_eaten(agent, game_state)
                    or game_state.is_over()
                    or self.is_eaten(opponent1, game_state)):
                return ['action?', evaluation(game_state)]

            else:
                # make list [[action, value],[action,value], ...]
                scored_actions = []
                actions = game_state.get_legal_actions(agent)
                v = 0
                if agent == caller:
                    v = -v
                for action in actions:
                    new_state = game_state.generate_successor(agent, action)
                    if agent == caller:
                        minimax_max_value = mini_max(new_state, opponent1, depth_minimax, a, b)[1]
                        v = max(v, minimax_max_value)  # pseudocode van alpha beta
                        if v > b:
                            return [action, v]
                        else:
                            a = max(a, v)
                        scored_actions.append([action, minimax_max_value])
                    else:
                        if opponent2 and agent == opponent2:
                            minimax_min_value = mini_max(new_state, caller, depth_minimax - 1, a, b)[1]
                        elif opponent2 and agent == opponent1:
                            minimax_min_value = mini_max(new_state, opponent2, depth_minimax, a, b)[1]
                        elif not opponent2 and agent == opponent1:
                            minimax_min_value = mini_max(new_state, caller, depth_minimax - 1, a, b)[1]
                        else:
                            return util.raiseNotDefined()

                        v = min(v, minimax_min_value)
                        if v < a:
                            return [action, v]
                        else:
                            b = min(b, v)
                        scored_actions.append([action, minimax_min_value])

                # choose best from list
                best_scored_action = scored_actions[0]
                for scored_action in scored_actions:
                    if agent == caller:
                        if scored_action[1] > best_scored_action[1]:
                            best_scored_action = scored_action
                    if agent != caller:
                        if scored_action[1] < best_scored_action[1]:
                            best_scored_action = scored_action
                return best_scored_action

        return_value = mini_max(game_state, caller, depth, -float('inf'), float('inf'))[0]
        self.minimax_depth = old_minimax_depth
        return return_value


class OffensiveAgent(DefensiveAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.max_food_to_get = 3
        self.aggressive_distance = 2
        self.status = 'heading_to_border'

    def choose_action(self, game_state):
        """
        chase or patrol
        """
        # data fill in
        agent_state = game_state.get_agent_state(self.index)
        curr_pos = agent_state.get_position()
        self.update_observable_opponents_and_distances(game_state)

        # JUST BRING IT IN
        if agent_state.num_carrying >= 1:
            for action in game_state.get_legal_actions(self.index):
                if not game_state.generate_successor(self.index, action).get_agent_state(self.index).is_pacman:
                    return action

        # CHASE
        border_distance = self.min_distance_to_own_side(game_state, curr_pos)
        if (border_distance > self.aggressive_distance
                and not agent_state.is_pacman
                and ((self.observable_opponents[0]
                      and util.manhattanDistance(curr_pos, self.observable_opponents[0]) <= 5)
                     or (self.observable_opponents[1]
                         and util.manhattanDistance(curr_pos, self.observable_opponents[1]) <= 5))):
            return self.alpha_beta_minimax(game_state, self.minimax_depth, self.evaluation_function_defence)
        # MINIMAX ATTACK
        elif (len(self.get_food(game_state).as_list()) >= 2
              and (border_distance <= self.aggressive_distance or agent_state.is_pacman)
              and ((self.observable_opponents[0]
                    and util.manhattanDistance(curr_pos, self.observable_opponents[0]) <= 5)
                   or (self.observable_opponents[1]
                       and util.manhattanDistance(curr_pos, self.observable_opponents[1]) <= 5))):
            return self.alpha_beta_minimax(game_state, self.minimax_depth, self.evaluation_function_offence)

        # ATTACK
        elif (border_distance <= 2 or agent_state.is_pacman) <= len(
                self.get_food(game_state).as_list()) and agent_state.num_carrying <= self.max_food_to_get:
            actions = game_state.get_legal_actions(self.index)
            best_dist = 999
            best_action = 999
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                food_list = self.get_food(game_state).as_list()
                closest_food = food_list[0]
                for food in food_list:
                    if self.get_maze_distance(closest_food, curr_pos) > self.get_maze_distance(food, curr_pos):
                        closest_food = food
                dist = self.get_maze_distance(closest_food, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        # HEAD TO BORDER
        else:  # head to border
            actions = game_state.get_legal_actions(self.index)
            best_dist = 999
            best_action = 999
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.min_distance_to_own_side(successor, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

    def evaluation_function_offence(self, game_state):
        """
                defensive evaluation function, expanded on DefensiveReflexAgent
                """
        features = util.Counter()
        agent_state = game_state.get_agent_state(self.index)
        curr_pos = agent_state.get_position()
        self.update_observable_opponents_and_distances(game_state)

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if agent_state.is_pacman:
            features['on_defense'] = 0

        # Computes distance to invaders we can see
        defenders = []
        for ind, pos in enumerate(self.observable_opponents):
            if pos and (ind == 0) and game_state.get_agent_state(self.opponents[0]).scared_timer <= 0:
                defenders.append(self.opponents[0])
            elif pos and ind == 1 and game_state.get_agent_state(self.opponents[1]).scared_timer <= 0:
                defenders.append(self.opponents[1])
        features['num_defenders'] = len(defenders)
        if len(defenders) > 0:
            features['defender_distance'] = min(self.distances_opponents)

        # what to do when scared_timer is active?
        features['scared_timer'] = agent_state.scared_timer

        # factor in the score
        features['score'] = game_state.get_score()

        # defended food left
        food_list = self.get_food(game_state).as_list()
        features['food_left'] = len(food_list)
        features['dis_nearest_food'] = min([self.get_maze_distance(curr_pos, food) for food in food_list])

        features['is_eaten'] = self.is_eaten(self.index, game_state)

        features['num_carrying'] = agent_state.num_carrying

        features['num_returned'] = agent_state.num_returned

        features['capsules_left'] = len(game_state.get_capsules())

        if features['num_carrying'] >= 1 and features['num_defenders'] >= 1:
            features['on_defence'] = -1
            features['food_left'] = 0
            features['num_carrying'] = 0

        if features['num_carrying'] >= self.max_food_to_get:
            features['on_defence'] = -1
            features['distance_to_own_side'] = self.min_distance_to_own_side(game_state, curr_pos)

        weights = {'on_defense': -1, 'num_defenders': -1, 'defender_distance': 10, 'scared_timer': 0,
                   'score': 200, 'food_left': -1, 'is_eaten': -1000, 'num_carrying': 1, 'num_returned': 10,
                   'capsules_left': -10, 'distance_to_own_side': -10}
        return features * weights

    def min_distance_to_own_side(self, game_state, pos):
        if self.red:
            x = int(self.border)
        else:
            x = int(self.border + 1)
        positions = [(x, y) for y in range(1, self.height + 1) if not game_state.has_wall(x, y)]
        if not positions:
            return util.raiseNotDefined
        else:
            distance_to_home = min(self.get_maze_distance(position, pos) for position in positions)
            return distance_to_home
