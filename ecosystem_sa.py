import random
import cv2
import os
import math
from copy import copy
import functools

import numpy as np
import gymnasium
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import wrappers, parallel_to_aec

def env(render_mode=None):
    env = raw_env(render_mode=render_mode)
    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-9999)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(render_mode=None):
     env = parallel_env(render_mode=render_mode)
     env = parallel_to_aec(env)
     return env


class parallel_env(ParallelEnv):
    metadata = {
        "name": "ecosystem_v2",
        "render_modes": ["human", "none"],
        "is_parallelizable": False,
        "render_fps": 2
    }

    def __init__(self, render_mode, observation_mode="single", params={}, image_path=""):
        super().__init__()
        self.render_mode = render_mode
        self.observation_mode = observation_mode

        self.params = {
            "mapsize": 128,
            "min_depth": 10,
            "max_depth": 30,
            "starting_population": 15,
            "food_value": 200,
            "movement_cost": -1,
            "death_penalty": -2000,
            "illegal": -20,
            "positive_food_mult": 2,
            "food_per_agent": 4,
            "max_timesteps": 200000,
            "distance_factor": 3,
            "eyesight_range": 32,
            "visible_agents": 1,
            "visible_foods": 2,
            "max_food": 400,
            "max_hp": 1000,
            "bite_damage": 400,
            "feed_range": 3,
            "bite_range": 3,
            "dmg_penalty_mult": 1,
        }
        for key in params:
            if key in self.params:
                self.params[key] = params[key]
            else:
                print("Parameter", key, "is invalid")

        self.mapsize = (self.params["mapsize"], self.params["mapsize"])
        self.min_depth = 10
        self.max_depth = 30
        self.visible_agent_amount = self.params["visible_agents"]
        self.visible_food_amount = self.params["visible_foods"]
        print("Loading map image")
        if image_path == "":
            self.filename = os.path.join(
                os.path.dirname(__file__), 'adv_map.png')
        else:
            self.filename = image_path
        print("Map image loaded")
        self.depth_map = cv2.imread(self.filename)
        self.depth_map = cv2.cvtColor(self.depth_map, cv2.COLOR_BGR2GRAY)
        self.depth_map = cv2.resize(
            self.depth_map, self.mapsize, interpolation=cv2.INTER_CUBIC)
        self.cross = math.sqrt(
            self.mapsize[0] ** 2 + self.mapsize[1] ** 2 + self.max_depth ** 2)

        # Agent properties

        self.foodAmount = self.params["starting_population"] * self.params["food_per_agent"]
        self.sight_distance = self.params["eyesight_range"]

        self.max_food = self.params["max_food"]
        self.max_hp = self.params["max_hp"]
        self.feedRange = self.params["feed_range"]
        self.bite_damage = self.params["bite_damage"]
        self.attackRange = self.params["bite_range"]

        self.possible_agents = [f"agent_{i}" for i in range(
            self.params["starting_population"])]
        self.foods = []

        self.timestep = 0  # Resets the timesteps
        if self.observation_mode == "single":
            self.terrain_shape = (3, 3)
        elif self.observation_mode == "image":
            self.terrain_shape = (self.sight_distance * 2 + 1, self.sight_distance * 2 + 1)
        else:
            self.terrain_shape = (self.sight_distance * 2 + 1, self.sight_distance * 2 + 1)
        self.r_x = int((self.terrain_shape[0] - 1) / 2)
        self.r_y = int((self.terrain_shape[0] - 1) / 2)
        
        metadata = {"render_modes": ["human"], "name": "ecosystem_v2"}

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if self.observation_mode == "single":
            return spaces.Box(-1.0, 1.0, (5 + 2 + self.visible_agent_amount * 6 + self.visible_food_amount * 6 + 7,), np.float32)
        else:
            return spaces.Dict({
                "action_mask": spaces.Box(low=-1., high=1., shape=(7,), dtype=np.float32),
                "terrain": spaces.Box(low=-1., high=1., shape=self.terrain_shape, dtype=np.float32),
                "self_data": spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32),
                "foods": spaces.Box(low=-1., high=1., shape=(self.visible_food_amount, 3), dtype=np.float32),
                "agents": spaces.Box(low=-1., high=1., shape=(self.visible_agent_amount, 3), dtype=np.float32)
            })
        
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(7)
    
    def action_mask():
        return spaces.Box(0, 1, (7,), dtype=int)

    def render(self):  # Simple OpenCV display of the environment
        image = self.toImage((400, 400))
        scale_x = 400 / self.mapsize[0]
        scale_y = 400 / self.mapsize[1]
        color = (0, 0, 255)
        for agent in self.agents:
            org = (int(self.agentData[agent]["y"] * scale_x),
                   int(self.agentData[agent]["x"] * scale_y - 10))
            image = cv2.putText(image, str(self.agentData[agent]["depth"]), org, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)
        cv2.imshow("map", image)
        cv2.waitKey(1)

    def toImage(self, window_size):  # Converts the map to a ready to display image
        agent_color = [0, 0, 255]
        food_color = [0, 255, 0]

        img = cv2.bitwise_not(self.display_map)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for food in self.foods:
            img[food[0]][food[1]][0] = food_color[0]
            img[food[0]][food[1]][1] = food_color[1]
            img[food[0]][food[1]][2] = food_color[2]

        for agent in self.agents:
            a_x = self.agentData[agent]["x"]
            a_y = self.agentData[agent]["y"]
            a_d = self.agentData[agent]["depth"]
            color_factor = 0.25 + (1 - (a_d / self.max_depth)) * 0.75
            img[a_x][a_y][0] = agent_color[0] * color_factor
            img[a_x][a_y][1] = agent_color[1] * color_factor
            img[a_x][a_y][2] = agent_color[2] * color_factor

        return cv2.resize(img, window_size, interpolation=cv2.INTER_NEAREST)

    def close(self):
        print(self.timestep)
        cv2.destroyAllWindows()

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.agentData = {}
        self.timestep = 0
        self.map = np.zeros(self.mapsize)
        self.foods = []
        # This function starts the world:
        # loads the map from image,
        # spawns agents and generates food
        self.generateMap()
        self.terminated = []
        self.rewards = {
            agentID: 0.0 for agentID in self.agents
        }
        observations = {agent: self.observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.state = observations

        return observations, infos

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        self.rewards = {agentID: 0.0 for agentID in self.agents}
        eaters = []
        rest = {}
        for agent, action in actions.items():
            if agent in self.agents:
                if action == 6:
                    eaters.append(agent)
                else:
                    rest[agent] = action

        for agent in eaters:
            food = self.performAction(agent, 6)
            if food > 0:
                self.rewards[agent] += food * self.params["positive_food_mult"]
            else:
                self.rewards[agent] += food

        for agent, action in rest.items():
            food = self.performAction(agent, action)
            self.rewards[agent] += food

        self.timestep += 1
        env_truncation = self.timestep >= self.params["max_timesteps"]
        truncations = {agent: env_truncation for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        
        for agent in self.agents:
            if self.agentData[agent]["hp"] <= 0 or self.agentData[agent]["food"] <= 0:
                self.agents.pop(self.agents.index(agent))
                terminations[agent] = True
                self.rewards[agent] += self.params["death_penalty"]

        observations = {
            agent: self.observation(agent, actions[agent]) for agent in self.agents
        }
        self.state = observations
        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        
        return observations, self.rewards, terminations, truncations, infos
    
    def observation_image(self, agentID):
        a_x = self.agentData[agentID]["x"]
        a_y = self.agentData[agentID]["y"]
        a_d = self.agentData[agentID]["depth"]
        offset = self.sight_distance
        terrain = np.zeros(shape=(self.terrain_shape[0], self.terrain_shape[1], 3), dtype=np.uint8)
        for x in range(self.terrain_shape[0]):
            for y in range(self.terrain_shape[1]):
                if a_x - x >= 0 and a_y - y >= 0:
                    pixel_value = int((self.map[a_x - x - offset][a_y - y - offset] - a_d) / self.max_depth * 255)
                else:
                    pixel_value = 255
                terrain[y][x][0] = pixel_value
                terrain[y][x][1] = pixel_value
                terrain[y][x][2] = pixel_value
                # terrain[x][y][0] = self.depth_map[a_x - x - offset][a_y - y - offset]
                # terrain[x][y][1] = self.depth_map[a_x - x - offset][a_y - y - offset]
                # terrain[x][y][2] = self.depth_map[a_x - x - offset][a_y - y - offset]

        for food in self.foods:
            if self.getDistance(a_x, a_y, a_d, food[0], food[1], food[2]) <= self.sight_distance / 2:
                terrain[a_x - food[0] + offset][a_y - food[1] + offset][0] = 0
                terrain[a_x - food[0] + offset][a_y - food[1] + offset][1] = 255
                terrain[a_x - food[0] + offset][a_y - food[1] + offset][2] = 0
        
        for agent in self.agents:
            if agent != agentID and self.getDistance(a_x, a_y, a_d, self.agentData[agent]["x"], self.agentData[agent]["y"], self.agentData[agent]["depth"]) <= self.sight_distance / 2:
                terrain[a_x - self.agentData[agent]["x"] + offset][a_y - self.agentData[agent]["y"] + offset][0] = 0
                terrain[a_x - self.agentData[agent]["x"] + offset][a_y - self.agentData[agent]["y"] + offset][1] = 0
                terrain[a_x - self.agentData[agent]["x"] + offset][a_y - self.agentData[agent]["y"] + offset][2] = 255
        
        return terrain
    
    def observation(self, agentID, action=None):
        a_x = self.agentData[agentID]["x"]
        a_y = self.agentData[agentID]["y"]
        a_d = self.agentData[agentID]["depth"]
        visibleAgents = []
        visibleFoods = []

        terrain = np.full(self.terrain_shape, 1.0)
        surrounding = [1.0, 1.0, 1.0, 1.0, 1.0]

        for x in range(-self.r_x, self.r_x + 1):
            for y in range(-self.r_y, self.r_y + 1):
                c_x = a_x + x
                c_y = a_y + y
                if 0 <= c_x < self.mapsize[0] and 0 <= c_y < self.mapsize[1]:
                    terrain[x][y] = (
                        self.map[c_x][c_y] - a_d
                        ) / self.max_depth * 2 - 1

        surrounding[0] = terrain[2][1]
        surrounding[1] = terrain[0][1]
        surrounding[2] = terrain[1][1]
        surrounding[3] = terrain[1][2]
        surrounding[4] = terrain[1][0]

        for agent in self.agents:
            if agentID != agent:
                b_x = self.agentData[agent]["x"]
                b_y = self.agentData[agent]["y"]
                b_d = self.agentData[agent]["depth"]
                if self.rayCast(a_x, a_y, a_d, b_x, b_y, b_d):
                    x, y, d, dist, best_action = self.calcVector(
                        agentID, agent)
                    visibleAgents.append([dist, best_action, -1, x, y, d])
        
        agents = sorted(visibleAgents, key=lambda x: x[0])[:self.visible_agent_amount]
        if len(agents) > 0:
            closest_agent = agents[0][0]
        else:
            closest_agent = self.sight_distance
        for a in agents:
            a[0] = a[0] / self.sight_distance * 2 - 1
            a[1] = a[1] / 3 - 1
            a[3] = a[3] / self.sight_distance * 2 - 1
            a[4] = a[4] / self.sight_distance * 2 - 1
            a[5] = a[5] / self.sight_distance * 2 - 1
        if len(agents) < self.visible_agent_amount:
            for i in range(0, self.visible_agent_amount - len(agents)):
                agents.append([0, 0, 0, 0, 0, 0])
        
        for food in self.foods:
            x, y, d, dist, best_action = self.calcVectorCoord(
                agentID, food[0], food[1], food[2])
            if dist < self.sight_distance:
                visibleFoods.append([dist, best_action, 1, x, y, d])

        foods = sorted(visibleFoods, key=lambda x: x[0])[:self.visible_food_amount]
        if len(foods) > 0:
            closest_food = foods[0][0]
        else:
            closest_food = self.sight_distance
        for f in foods:
            f[0] = f[0] / self.sight_distance * 2 - 1
            f[1] = f[1] / 3 - 1
            f[3] = f[3] / self.sight_distance * 2 - 1
            f[4] = f[4] / self.sight_distance * 2 - 1
            f[5] = f[5] / self.sight_distance * 2 - 1
        if len(foods) < self.visible_food_amount:
            for i in range(0, self.visible_food_amount - len(foods)):
                foods.append([0, 0, 0, 0, 0, 0])
        
        b_x = self.agentData[agentID]["x"] / self.cross * 2 - 1
        b_y = self.agentData[agentID]["y"] / self.cross * 2 - 1
        b_d = self.agentData[agentID]["depth"] / self.cross * 2 - 1
        hp = self.agentData[agentID]["hp"] / self.max_hp * 2 - 1
        food = self.agentData[agentID]["food"] / self.max_food * 2 - 1
        if food > 1:
            food = 1.0
            
        self_info = [hp, food]

        closest = []
        for x in agents:
            closest += x
        for x in foods:
            closest += x

        distance_reward = (1 - (closest_food / self.sight_distance)) ** 2 * self.params["distance_factor"]
        self.rewards[agentID] += distance_reward

        action_mask = np.ones(7)
        if a_y >= self.mapsize[1] - 1:
            action_mask[0] = -1
        elif a_y <= 0:
            action_mask[1] = -1
        if a_x >= self.mapsize[0] - 1:
            action_mask[2] = -1
        elif a_x <= 0:
            action_mask[3] = -1
        if a_d >= self.map[a_x][a_y] - 1:
            action_mask[4] = -1
        elif a_d <= 0:
            action_mask[5] = -1
        if closest_food > self.feedRange and closest_agent > self.attackRange:
            action_mask[6] = -1

        # Currently the flattening of observation vectors is done here
        # 2 + 5 + 3*5 + 7 = 2 + 5 + 15 + 7 = 29
        if self.observation_mode.lower() == "single":
            observation =  self_info + surrounding + closest + action_mask.tolist()
            return np.array(observation)
        else:
            observation = {
                "action_mask": np.array(action_mask),
                "terrain": terrain,
                "self_data": np.array(self_info),
                "foods": np.array(foods),
                "agents": np.array(agents)
            }
            return observation

    
    # This function either performs or calculates the effect of performing a certain action
    def performAction(self, agentID, action):
        if self.agentData[agentID]["hp"] <= 0 or self.agentData[agentID]["food"] <= 0:
            return 0
        x = self.agentData[agentID]["x"]
        y = self.agentData[agentID]["y"]
        d = self.agentData[agentID]["depth"]
        f = 0

        match action:
            case 0:  # y+
                if y < self.mapsize[1] - 1 and self.depth_map[x][y + 1] > d:
                    self.agentData[agentID]["y"] += 1
                    f += self.params["movement_cost"]
            case 1:  # y-
                if y > 0 and self.depth_map[x][y - 1] > d:
                    self.agentData[agentID]["y"] += -1
                    f += self.params["movement_cost"]
            case 2:  # x+
                if x < self.mapsize[0] - 1 and self.depth_map[x + 1][y] > d:
                    self.agentData[agentID]["x"] += 1
                    f += self.params["movement_cost"]
            case 3:  # x-
                if x > 0 and self.depth_map[x - 1][y] > d:
                    self.agentData[agentID]["x"] += -1
                    f += self.params["movement_cost"]
            case 4:  # depth+
                if d < self.map[x][y] - 1:
                    self.agentData[agentID]["depth"] += 1
                    f += self.params["movement_cost"]
            case 5:  # depth-
                if d > 0:
                    self.agentData[agentID]["depth"] += -1
                    f += self.params["movement_cost"]
            case 6:
                eaten = False
                for agent in self.agents:
                    if agentID != agent:
                        if self.getDistance(
                            x, y, d, self.agentData[agent]["x"],
                            self.agentData[agent]["y"],
                            self.agentData[agent]["depth"]
                            ) <= self.attackRange and eaten == False:
                            self.damage(agent)
                            f += self.bite_damage
                            eaten = True
                            break
                if eaten == False:
                    for food in self.foods:
                        if self.getDistance(
                            x, y, d, food[0], food[1], food[2]
                            ) <= self.feedRange and eaten == False:
                            f += self.params["food_value"]
                            self.foods.pop(self.foods.index(food))
                            eaten = True
                            self.agentData[agentID]["hp"] = self.max_hp
                            self.generateNewFood()
                            break
        if f == 0:
            self.rewards[agentID] += self.params["illegal"]
            f += self.params["movement_cost"]

        self.agentData[agentID]["food"] += f
        return f
    
    def damage(self, agentID):
        self.agentData[agentID]["hp"] -= self.bite_damage
        self.rewards[agentID] -= self.bite_damage * \
            self.params["dmg_penalty_mult"]

    def calcVector(self, agent1, agent2):
        x = self.agentData[agent2]["x"] - self.agentData[agent1]["x"]
        y = self.agentData[agent2]["y"] - self.agentData[agent1]["y"]
        d = self.agentData[agent2]["depth"] - self.agentData[agent1]["depth"]
        distance = self.getDistance(self.agentData[agent1]["x"],
                                    self.agentData[agent1]["y"],
                                    self.agentData[agent1]["depth"],
                                    self.agentData[agent2]["x"],
                                    self.agentData[agent2]["y"],
                                    self.agentData[agent2]["depth"])
        return x, y, d, distance, self.bestActionAttack(x, y, d, distance)
    
    def calcVectorCoord(self, agentID, b_x, b_y, b_d):
        x = b_x - self.agentData[agentID]["x"]
        y = b_y - self.agentData[agentID]["y"]
        d = b_d - self.agentData[agentID]["depth"]
        distance = self.getDistance(self.agentData[agentID]["x"],
                                    self.agentData[agentID]["y"],
                                    self.agentData[agentID]["depth"],
                                    b_x,
                                    b_y,
                                    b_d)
        return x, y, d, distance, self.bestAction(x, y, d, distance)
    
    # Function to select what action should be taken to go towards coordinates
    def bestAction(self, x, y, d, dist):
        if dist < self.feedRange:
            return 6
        if abs(y) >= abs(x) and abs(y) >= abs(d):
            if y > 0:
                return 0
            else:
                return 1
        if abs(x) >= abs(y) and abs(x) >= abs(d):
            if x > 0:
                return 2
            else:
                return 3
        if abs(d) >= abs(y) and abs(d) >= abs(x):
            if d > 0:
                return 4
            else:
                return 5
            
    def bestActionAttack(self, x, y, d, dist):
        if dist < self.attackRange:
            return 6
        if abs(y) >= abs(x) and abs(y) >= abs(d):
            if y > 0:
                return 0
            else:
                return 1
        if abs(x) >= abs(y) and abs(x) >= abs(d):
            if x > 0:
                return 2
            else:
                return 3
        if abs(d) >= abs(y) and abs(d) >= abs(x):
            if d > 0:
                return 4
            else:
                return 5
            
    def getDistance(self, a_x, a_y, a_d, b_x, b_y, b_d):
        return math.sqrt((b_x - a_x) ** 2 + (b_y - a_y) ** 2 + (b_d - a_d) ** 2)

############################################################################################################

    def cstCoord(self, x, y):  # Constrain the passed coordinates so they don't exceed the map
        if x > self.mapsize[0] - 1:
            x = self.mapsize[0] - 1
        elif x < 0:
            x = 0
        if y > self.mapsize[1] - 1:
            y = self.mapsize[1] - 1
        elif y < 0:
            y = 0
        return x, y

############################################################################################################

    def gridRInt(self, xy):  # Returns random int in the map axis limit
        if xy == "y":
            return random.randint(0, self.mapsize[0] - 1)
        else:
            return random.randint(0, self.mapsize[1] - 1)
        
    def generateMap(self):
        # Loads map file and converts it to a discrete terrain map
        self.map = ((self.max_depth - self.min_depth)*(self.depth_map -
                                                       np.min(self.depth_map))/np.ptp(self.depth_map)).astype(int) + self.min_depth
        self.display_map = ((self.depth_map - self.depth_map.min()) * (1/(self.depth_map.max() - self.depth_map.min()) * 255)).astype('uint8')

        self.generateNewFood()

        # Generate agents
        for agentID in self.agents:
            x = self.gridRInt("x")
            y = self.gridRInt("y")
            depth_point = self.map[x][y]
            depth = random.randint(0, depth_point - 1)
            self.createNewAgent(x, y, depth, agentID, False)

############################################################################################################

    # Check how much food is present and generate what is missing
    def generateNewFood(self):
        currentFood = len(self.foods)
        while currentFood < self.foodAmount:
            x = self.gridRInt("x")
            y = self.gridRInt("y")
            depth_point = self.map[x][y]
            self.foods.append([x, y, depth_point])
            currentFood = len(self.foods)

############################################################################################################
    # Check if 2 coordinates have line of sight
    def rayCast(self, a_x, a_y, a_d, b_x, b_y, b_d):
        dist = self.getDistance(a_x, a_y, a_d, b_x, b_y, b_d)
        if dist > self.sight_distance:
            return False
        if abs(dist) < 0.001:
            return True
        vec = [b_x - a_x, b_y - a_y, b_d - a_d]
        mini = [vec[0] / dist, vec[1] / dist, vec[2] / dist]
        c_x = a_x
        c_y = a_y
        c_d = a_d

        for i in range(round(dist) + 1):
            if self.map[round(c_x)][round(c_y)] < c_d:
                if round(c_x) == b_x and round(c_y) == b_y:
                    return True
                else:
                    return False
            c_x += mini[0]
            c_y += mini[1]
            c_d += mini[2]

            if b_x < 0:
                if c_x <= b_x:
                    return True
            else:
                if c_x >= b_x:
                    return True
            if b_y < 0:
                if c_y <= b_y:
                    return True
            else:
                if c_y >= b_y:
                    return True

        return True
    
    def createNewAgent(self, x, y, d, agentID, is_new):
        if is_new:
            self.agents.append(agentID)
            self.rewards[agentID] = 0.0
        hp = self.max_hp
        food = self.max_food

        self.agentData[agentID] = {
            "x": x, "y": y, "depth": d, "hp": hp, "food": food, "egg": self.timestep}