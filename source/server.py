import mesa
from matplotlib.cm import get_cmap
import numpy as np
from .model import PacmanGame, PacmanAgent, GhostAgent, AntAgent, CoinAgent, WallAgent, CellAgent

GRID_SIZE = 30
CANVAS_SIZE = 600

cmap = get_cmap('YlOrRd')

def rgba_to_hex(color):
    color = np.rint(np.array(color)[:3] * 255)
    return '#{:02X}{:02X}{:02X}'.format(int(color[0]), int(color[1]), int(color[2]))

def visualizer(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is PacmanAgent:
        portrayal["Shape"] = "img/pacman_%s%s.png" %(agent.orientation, '_dead' if not agent.alive else '')
        portrayal["scale"] = 0.9
        portrayal["Layer"] = 4
    elif type(agent) is GhostAgent:
        portrayal["Shape"] = "img/ghost_blue_%s.png" %(agent.orientation)
        portrayal["scale"] = 0.9
        portrayal["Layer"] = 3
    elif type(agent) is CoinAgent:
        portrayal["Shape"] = "img/coin.png"
        portrayal["scale"] = 0.9
        portrayal["Layer"] = 1
    elif type(agent) is AntAgent and agent.model.ants_visible:
        portrayal["Shape"] = "img/ant_%s.png" %(agent.orientation)
        portrayal["scale"] = 0.5
        portrayal["Layer"] = 2
    elif type(agent) is WallAgent:
        portrayal["Shape"] = "rect"
        portrayal["Color"] = "#000000"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["w"] = 1
        portrayal["h"] = 1
    elif type(agent) is CellAgent and agent.model.pheromones_visible:
        portrayal["Shape"] = "rect"
        portrayal["Color"] = rgba_to_hex(cmap(agent.tau / agent.model.tau_max))
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["w"] = 1
        portrayal["h"] = 1

    return portrayal

canvas_element = mesa.visualization.CanvasGrid(visualizer, GRID_SIZE, GRID_SIZE, CANVAS_SIZE, CANVAS_SIZE)

def collected_coins(model):
    return 'Number of collected coins: {}'.format(model.datacollector.model_reporters['Collected coins'](model))

model_params = {
    "grid_size": GRID_SIZE,
    "N_ghosts": mesa.visualization.Slider('Number of ghosts', 5, 1, 20),
    "N_ants": mesa.visualization.Slider('Number of ants', 50, 5, 100),
    "ants_visible": mesa.visualization.Checkbox('Ants visible', False, "Choose whether you want to see Pacman's path planners or not."),
    "pheromones_visible": mesa.visualization.Checkbox('Pheromone levels visible', False),
    "pheromone_evaporation_coefficient": 1.5e-1,
    "alpha": 3.,
    "beta": 4.
}

server = mesa.visualization.ModularServer(PacmanGame, [canvas_element, collected_coins], "Pacman", model_params)
server.port = 8521
