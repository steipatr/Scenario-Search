from mesa import Model as MesaModel
from mesa import Agent
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

import random

import numpy as np
    
#new version
# class SchellingAgent(Agent):
#     """
#     Schelling segregation agent
#     """

#     def __init__(self, pos, model, agent_type):
#         """
#         Create a new Schelling agent.

#         Args:
#            unique_id: Unique identifier for the agent.
#            x, y: Agent initial location.
#            agent_type: Indicator for the agent's type (minority=1, majority=0)
#         """
#         super().__init__(pos, model)
#         self.pos = pos
#         self.type = agent_type

#     def step(self):
#         similar = 0
#         for neighbor in self.model.grid.iter_neighbors(self.pos, True):
#             if neighbor.type == self.type:
#                 similar += 1

#         # If unhappy, move:
#         if similar < self.model.homophily:
#             self.model.grid.move_to_empty(self)
#         else:
#             self.model.happy += 1

# class Schelling(MesaModel):
#     """
#     Model class for the Schelling segregation model.
#     """

#     def __init__(self, width=20, height=20, density=0.8, minority_pc=0.2, homophily=3):
#         """ """

#         self.width = width
#         self.height = height
#         self.density = density
#         self.minority_pc = minority_pc
#         self.homophily = homophily

#         self.schedule = RandomActivation(self)
#         self.grid = SingleGrid(width, height, torus=True)

#         self.happy = 0
#         self.datacollector = DataCollector(
#             {"happy": "happy"},  # Model-level count of happy agents
#             # For testing purposes, agent's individual x and y
#             {"x": lambda a: a.pos[0], "y": lambda a: a.pos[1]},
#         )

#         # Set up agents
#         # We use a grid iterator that returns
#         # the coordinates of a cell as well as
#         # its contents. (coord_iter)
#         for cell in self.grid.coord_iter():
#             x, y = cell[1]
#             if self.random.random() < self.density:
#                 agent_type = 1 if self.random.random() < self.minority_pc else 0

#                 agent = SchellingAgent((x, y), self, agent_type)
#                 self.grid.place_agent(agent, (x, y))
#                 self.schedule.add(agent)

#         self.running = True
#         self.datacollector.collect(self)

#     def step(self):
#         """
#         Run one step of the model. If All agents are happy, halt the model.
#         """
#         self.happy = 0  # Reset counter of happy agents
#         self.schedule.step()
#         # collect data
#         self.datacollector.collect(self)

#         if self.happy == self.schedule.get_agent_count():
#             self.running = False
    
#old version
class SchellingAgent(Agent):
    '''
    Schelling segregation agent
    '''
    def __init__(self, pos, model, agent_type):
        '''
         Create a new Schelling agent.
         Args:
            unique_id: Unique identifier for the agent.
            x, y: Agent initial location.
            agent_type: Indicator for the agent's type (minority=1, majority=0)
        '''
        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type

    def step(self):
        similar = 0
        neighbors = self.model.grid.neighbor_iter(self.pos) #Moore neighborhood/Queen's move
        for neighbor in neighbors:
            if neighbor.type == self.type:
                similar += 1

        # If unhappy, move:
        if similar < self.model.homophily:
            self.model.grid.move_to_empty(self)
        else:
            self.model.happy += 1


class SchellingModel(MesaModel):
    '''
    Model class for the Schelling segregation model.
    '''

    def __init__(self, height, width, density, minority_pc, homophily):
        '''
        '''

        self.height = height
        self.width = width
        self.density = density
        self.minority_pc = minority_pc
        self.homophily = homophily

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(height, width, torus=True)

        self.happy = 0
        self.datacollector = DataCollector(
            {"happy": lambda m: m.happy},  # Model-level count of happy agents
            # For testing purposes, agent's individual x and y
            {"x": lambda a: a.pos[0], "y": lambda a: a.pos[1]})

        self.running = True

        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]
            if random.random() < self.density:
                if random.random() < self.minority_pc:
                    agent_type = 1
                else:
                    agent_type = 0

                agent = SchellingAgent((x, y), self, agent_type)
                self.grid.position_agent(agent, (x, y))
                self.schedule.add(agent)

    def step(self):
        '''
        Run one step of the model. If All agents are happy, halt the model.
        '''
        self.happy = 0  # Reset counter of happy agents
        self.schedule.step()
        self.datacollector.collect(self)
        if self.happy == self.schedule.get_agent_count():
            self.running = False
    

#wrap model for EMA Workbench
def schelling_wrapper(density, homophily, height=30, width=30, minority_pc=0.5, max_steps = 100):

    # try:
    #     model = Schelling(height, width, density, minority_pc, homophily)
    # except:
    model = SchellingModel(height, width, density, minority_pc, homophily)

    #run model
    while model.running and model.schedule.steps < max_steps:
        model.step()

    x = len(model.grid.grid)
    y = len(model.grid.grid[0])    

    #get happiness
    n_happy = model.happy
    n_agents = model.schedule.get_agent_count()
    happiness = float(n_happy / n_agents)

    #get grid arrangement
    grid_data = np.empty((x,y))

    #Mesa Schelling model uses some non-obvious codes for patch types
    for i in range(x):
        for q in range(y):
            try:
                  val = model.grid.grid[i][q].type
            except:
                  val = 42 #empty patches

            if val == 0: #species 1
                val = 1
            elif val == 1: #species 2
                val = 2
            elif val == 42: #empty patch
                val = 0

            grid_data[i,q] = val

    return {"grid":grid_data,
           'happiness':happiness,
           'n_steps':float(model.schedule.steps),
           'n_agents':float(n_agents)}