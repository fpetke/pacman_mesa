import mesa
import numpy as np
import random

def maze_neighbors(grid_size: int, pos):
    '''Returns the positions of the grid neighbors.'''

    i_up = pos[0] - 2
    i_down = pos[0] + 2
    j_left = pos[1] - 2
    j_right = pos[1] + 2

    if i_up < 0:
        i_up += grid_size
    elif i_down >= grid_size:
        i_down -= grid_size
    if j_left < 0:
        j_left += grid_size
    elif j_right >= grid_size:
        j_right -= grid_size
    
    return (i_up, pos[1]), (i_down, pos[1]), (pos[0], j_left), (pos[0], j_right)

def cell_between(grid_size: int, pos1, pos2):
    '''Returns the coordinates of the cell, which is between pos1 and pos2.'''

    i_smaller = pos1[0] if pos1[0] < pos2[0] else pos2[0]
    i_bigger = pos1[0] if pos1[0] > pos2[0] else pos2[0]
    j_smaller = pos1[1] if pos1[1] < pos2[1] else pos2[1]
    j_bigger = pos1[1] if pos1[1] > pos2[1] else pos2[1]

    i = ((i_smaller+i_bigger)//2 if i_bigger-i_smaller<=2 else (i_smaller+i_bigger+grid_size)//2)
    j = ((j_smaller+j_bigger)//2 if j_bigger-j_smaller<=2 else (j_smaller+j_bigger+grid_size)//2)

    if i >= grid_size:
        i -= grid_size
    if j >= grid_size:
        j -= grid_size

    return (i, j)

def generate_maze_walls(grid_size: int):
    '''Randomized Prim's algorithm'''
    maze = np.ones((grid_size, grid_size), np.bool_)
    frontiers = []

    pos = tuple(np.random.randint(0, grid_size, 2))
    maze[pos] = False
    for neighbor in maze_neighbors(grid_size, pos):
        frontiers.append(neighbor)
    
    while frontiers:
        frontier = random.choice(frontiers)
        frontiers.remove(frontier)
        neighbor_cells = []
        for neighbor in maze_neighbors(grid_size, frontier):
            if not maze[neighbor]:
                neighbor_cells.append(neighbor)
        
        if len(neighbor_cells) > 0:
            cell_to_connect = random.choice(neighbor_cells)
            maze[frontier] = False
            maze[cell_between(grid_size, frontier, cell_to_connect)] = False

            for neighbor in maze_neighbors(grid_size, frontier):
                if maze[neighbor]:
                    frontiers.append(neighbor)

    return maze

class PacmanGame(mesa.Model):
    """Pacman's goal is to collect as many coins as possible without meeting a ghost.
    One coin is available at a time, so Pacman's challange is to find that coin in the maze. The
    next coin appears upon collection. But Pacman doesn't have to do it alone: his loyal friends,
    the ants are very enthusiastic to help in finding the coin in the maze. But as Pacman doesn't
    speak the language of the ants, he is forced to follow their pheromone trails."""
    def __init__(self,
        grid_size: int = 50,
        N_ghosts: int = 5,
        N_ants: int = 10,
        ants_visible: bool = False,
        pheromones_visible: bool = False,
        pheromone_evaporation_coefficient = 1e-2,
        alpha = 1.,
        beta = 1.,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.grid = mesa.space.MultiGrid(self.grid_size, self.grid_size, True)
        self.schedule = mesa.time.RandomActivationByType(self)
        self.datacollector = mesa.DataCollector({'Collected coins': lambda m: m.pacman.coins})

        # create Wall and Cell
        maze = generate_maze_walls(self.grid_size)
        free_cells = []
        self.cells = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if maze[i, j]:
                    self.grid.place_agent(WallAgent(self.next_id(), self, (i, j)), (i, j))
                else:
                    free_cells.append((i, j))
                    cell = CellAgent(self.next_id(), self, (i, j))
                    self.cells[(i, j)] = cell
                    self.grid.place_agent(cell, (i, j))

        # create Pacman
        pos = random.choice(free_cells)
        free_cells.remove(pos)
        self.pacman = PacmanAgent(self.next_id(), self, pos)
        self.grid.place_agent(self.pacman, pos)
        self.schedule.add(self.pacman)

        # create Ghosts
        self.N_ghosts = N_ghosts
        for i in range(self.N_ghosts):
            pos = random.choice(free_cells)
            ghost = GhostAgent(self.next_id(), self, pos)
            self.grid.place_agent(ghost, pos)
            self.schedule.add(ghost)

        # create Coin
        pos = random.choice(free_cells)
        self.coin = CoinAgent(self.next_id(), self, pos)
        self.grid.place_agent(self.coin, pos)

        # create Ants
        self.N_ants = N_ants
        self.ants_visible = ants_visible
        for i in range(self.N_ants):
            ant = AntAgent(self.next_id(), self, self.pacman.pos)
            self.grid.place_agent(ant, self.pacman.pos)
            self.schedule.add(ant)

        self.tau_max = 1.
        self.rho = pheromone_evaporation_coefficient
        self.alpha = alpha
        self.beta = beta
        self.pheromones_visible = pheromones_visible

    def pheromone_update(self):
        for pos in self.cells:
            s = 0.
            for ant in self.cells[pos].ants_visited:
                if ant.found_coin:
                    s += ant.path[pos] / np.array(list(ant.path.values())).sum()
            self.cells[pos].tau = (1. - self.rho) * self.cells[pos].tau + s
            if self.cells[pos].tau > self.tau_max:
                self.tau_max = self.cells[pos].tau

    def coin_moved(self):
        self.tau_max = 1.
        for pos in self.cells:
            self.cells[pos].tau = 1e-5
            self.cells[pos].ants_visited.clear()

    def step(self):
        self.schedule.step_type(GhostAgent)

        self.schedule.step_type(AntAgent)
        self.pheromone_update()

        self.schedule.step_type(PacmanAgent)

    def run_model(self, n):
        pacman = list(self.schedule.agents_by_type[PacmanAgent].values())[0]
        i = 0
        while pacman.alive and i < n:
            self.step()
            i += 1

class RandomWalker(mesa.Agent):
    step_directions = {
        (-1, 0): 'left',
        (1, 0): 'right',
        (0, 1): 'up',
        (0, -1): 'down',
        (0, 0): None
    }

    def __init__(self, unique_id: int, model: mesa.Model, pos) -> None:
        super().__init__(unique_id, model)
        self.pos = pos
        self.last_pos = None
        self.steps = 0

        self.step_direction = self.random.choice(list(self.step_directions.keys()))
        while self.step_direction == (0, 0):
            self.step_direction = self.random.choice(list(self.step_directions.keys()))
        self.orientation = self.step_directions[self.step_direction]

    def calculate_cell_direction(self, pos, cell_pos):
        i = cell_pos[0] - pos[0]
        j = cell_pos[1] - pos[1]

        if i < -self.model.grid_size/2:
            i += self.model.grid_size
        elif i > self.model.grid_size/2:
            i -= self.model.grid_size
        if j < -self.model.grid_size/2:
            j += self.model.grid_size
        elif j > self.model.grid_size/2:
            j -= self.model.grid_size

        best_direction = (0, 0)
        for direction in self.step_directions:
            if np.dot(direction, (i, j)) > np.dot(best_direction, (i, j)):
                best_direction = direction
        return best_direction

    def update_direction(self, next_pos):
        self.step_direction = self.calculate_cell_direction(self.pos, next_pos)
        if self.step_directions[self.step_direction] in ['left', 'right'] and self.step_directions[self.step_direction] != self.orientation:
            self.orientation = self.step_directions[self.step_direction]

    def calculate_possible_steps(self):
        next_positions = []
        for pos in self.model.grid.get_neighborhood(self.pos, False, True):
            there_is_no_wall = True
            for cellmate in self.model.grid.get_cell_list_contents(pos):
                if type(cellmate) is WallAgent:
                    there_is_no_wall = False
            if there_is_no_wall:
                next_positions.append(pos)
        return next_positions

    def random_step(self):
        self.last_pos = self.pos
        possible_steps = self.calculate_possible_steps()
        next_pos = self.random.choice(possible_steps if len(possible_steps)>0 else [self.pos])
        self.update_direction(next_pos)
        self.model.grid.move_agent(self, next_pos)

class PacmanWalker(RandomWalker):
    def __init__(self, unique_id: int, model: mesa.Model, pos) -> None:
        super().__init__(unique_id, model, pos)
        self.path = {}

    def calculate_possible_steps(self):
        possible_positions = super().calculate_possible_steps()

        for cellmate in self.model.grid.get_cell_list_contents(self.pos):
            if type(cellmate) is GhostAgent:
                if self.pos in possible_positions:
                    possible_positions.remove(self.pos)
                if (cellmate.pos[0]-cellmate.step_direction[0], cellmate.pos[1]-cellmate.step_direction[1]) in possible_positions:
                    possible_positions.remove((cellmate.pos[0]-cellmate.step_direction[0], cellmate.pos[1]-cellmate.step_direction[1]))

        neighbors = {}
        for neighbor_cell in self.model.grid.get_neighborhood(self.pos, False, False, 2):
            neighbors[neighbor_cell] = self.model.grid.get_cell_list_contents(neighbor_cell)

        next_steps = []
        for possible_position in possible_positions:
            is_possible = True
            for neighbor_cell in neighbors:
                for cell_content in neighbors[neighbor_cell]:
                    if type(cell_content) is GhostAgent and (possible_position == neighbor_cell or self.calculate_cell_direction(self.pos, possible_position) == self.calculate_cell_direction(possible_position, neighbor_cell)):
                        is_possible = False
            if is_possible:
                next_steps.append(possible_position)
        return next_steps

    def weighted_choice(self, arr, weights):
        return arr[np.where(weights.cumsum() >= np.random.rand())[0][0]]

    def random_step(self):
        possible_steps = self.calculate_possible_steps()
        probabilities = np.zeros(len(possible_steps))
        for i in range(len(possible_steps)):
            eta = 1. / self.path[possible_steps[i]] if possible_steps[i] in self.path else 1.
            probabilities[i] = self.model.cells[possible_steps[i]].tau**self.model.alpha * eta**self.model.beta

        if np.any(probabilities > 0.):
            probabilities /= probabilities.sum()
            next_pos = self.weighted_choice(possible_steps, probabilities)
        else:
            next_pos = self.random.choice(possible_steps if len(possible_steps)>0 else [self.pos])

        self.update_direction(next_pos)
        self.model.grid.move_agent(self, next_pos)

        if next_pos in self.path:
            self.path[next_pos] += 1
        else:
            self.path[next_pos] = 1

class PacmanAgent(PacmanWalker):
    def __init__(self, unique_id: int, model: mesa.Model, pos) -> None:
        super().__init__(unique_id, model, pos)
        self.coins = 0
        self.alive = True
        self.ghosts_met = []
        if not self.orientation in ['left', 'right']:
            self.orientation = self.random.choice(['left', 'right'])

    def check_cell(self):
        cellmates = self.model.grid.get_cell_list_contents(self.pos)
        coin_collected = False
        if len(cellmates) > 1:
            for cellmate in cellmates:
                if type(cellmate) is GhostAgent:
                    self.alive = False
                    if cellmate.steps > self.steps:
                        self.ghosts_met.append(cellmate)
                elif type(cellmate) is CoinAgent:
                    coin_collected = True
                    cellmate.collected = True
                elif type(cellmate) is WallAgent:
                    self.alive = False
        if coin_collected and self.alive:
            self.coins += 1

            new_coin_pos = self.random.choice(list(self.model.cells.keys()))
            while new_coin_pos == self.pos:
                new_coin_pos = self.random.choice(list(self.model.cells.keys()))
            self.model.grid.move_agent(self.model.coin, new_coin_pos)

            self.path.clear()
            self.model.coin_moved()
            for ant in self.model.schedule.agents_by_type[AntAgent].values():
                self.model.grid.move_agent(ant, self.pos)
                ant.coin_moved()

    def step(self):
        if self.alive:
            self.check_cell()

            self.random_step()
            self.steps += 1

            if not self.alive:
                alive = True
                for ghost in self.ghosts_met:
                    if self.pos == ghost.last_pos:
                        alive = False
                self.ghosts_met.clear()
                self.alive = alive

            self.check_cell()

class AntAgent(PacmanWalker):
    def __init__(self, unique_id: int, model: mesa.Model, pos) -> None:
        super().__init__(unique_id, model, pos)
        self.found_coin = False

    def pheromone_deposition(self):
        if not self in self.model.cells[self.pos].ants_visited:
            self.model.cells[self.pos].ants_visited.append(self)
        if self.model.coin.pos == self.pos:
            self.found_coin = True

    def coin_moved(self):
        self.found_coin = False
        self.path.clear()

    def step(self):
        if self.model.pacman.alive:
            self.random_step()
            self.pheromone_deposition()
            self.steps += 1

    def update_direction(self, next_pos):
        self.step_direction = self.calculate_cell_direction(self.pos, next_pos)
        if self.step_directions[self.step_direction]:
            self.orientation = self.step_directions[self.step_direction]

class GhostAgent(RandomWalker):
    def __init__(self, unique_id: int, model: mesa.Model, pos) -> None:
        super().__init__(unique_id, model, pos)
        if not self.orientation in ['left', 'right']:
            self.orientation = self.random.choice(['left', 'right'])
    
    def step(self):
        if self.model.pacman.alive:
            self.random_step()
            self.steps += 1

class CoinAgent(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model, pos) -> None:
        super().__init__(unique_id, model)
        self.pos = pos

class WallAgent(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model, pos) -> None:
        super().__init__(unique_id, model)
        self.pos = pos

class CellAgent(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model, pos) -> None:
        super().__init__(unique_id, model)
        self.pos = pos
        self.tau = 1e-5
        self.ants_visited = []
