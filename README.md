# Pacman
by Péter Forgács - [Lakeside Labs](https://www.lakeside-labs.com/)

## Instructions to run the code

- Needed *__Python__ packages*:
    - `mesa`
    - `numpy`
    - `matplotlib`
- The following **command** in a *command-line interface (like Terminal, Windows PowerShell)* will install the packages:
```bash
pip install mesa numpy matplotlib
```
- After having the packages installed, the following **command** in a *command-line editor (like Terminal, Windows PowerShell)* will run the code:
```bash
mesa runserver
```
- After the previous command, the visual interface of the simulation is **supposed to open automatically in a browser**. In case of not opening automatically, just copy the link you can see in the *command-line interface* (for example: *Interface starting at http://127.0.0.1:8521*)
    - In the browser, after the simulation interface is opened, in the upper right corner there is a **Start** button to **start the simulation**, and a **Step** button to **propagate the simulation step-by-step**.

## Changing parameters

It is possible to change some parameters of the simulation in the browser, where the simulation is being visualized. **After changing a parameter** other than the *Frames Per Second* parameter, a **Reset** is needed.

In the `server.py` file it is possible to change the **GRID_SIZE** parameter to **modify the number of rows and columns** of the canvas

Have fun!

## About the model

An Ant Colony Optimization (ACO) algorithm is used to find the coin.

The pheromone level of a cell $\tau_{ij}$ are updated in the following way:

$$
\tau_{ij} = (1 - \rho) \cdot \tau_{ij} + \sum_k \frac{n_k^{ij}}{n_k}
$$

, where $\rho \in \left[ 0, 1 \right]$ is the pheromone evaporation coefficient, $k$ goes through each ant, that already found the coin, $n_k$ is the length of ant $k$'s path, and $n_k^{ij}$ is the number of times the ant $k$ stepped on cell $\left( i, j \right)$.

Now based on the pheromone levels, the possible cells for the next step of an ant and Pacman will have different probabilities. The probability of the next possible cell $p_{ij}^{\text{ant/Pacman}}$ is calculated in the following way:

$$
p_{ij}^{\text{ant/Pacman}} = \frac{\tau_{ij}^\alpha \cdot \eta_{ij}^\beta}{\sum_{\left( i^\prime, j^\prime \right)} \tau_{i^\prime j^\prime}^\alpha \cdot \eta_{i^\prime j^\prime}^\beta}
$$

, where $\alpha$ controls the influence of the pheromone levels, $\eta_{ij} = \left( n_{k\text{ or Pacman}}^{ij} \right)^{-1}$ is the desirability of cell $\left( i, j \right)$, $\beta$ controls the influence of the desirability, the coordinates $\left( i^\prime, j^\prime \right)$ are all the possible coordinates for the next step.
