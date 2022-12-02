# Pacman
by Péter Forgács

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
