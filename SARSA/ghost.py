import numpy as np

class Ghost:
    """Random ghost"""

    def __init__(self, x , y, seed):
        
        self.x = x
        self.y = y
        self.rng = np.random.default_rng(seed=seed)
        
    
    def act(self, maze):
        eligible_actions = maze.eligible_actions(self.x, self.y)
        return self.rng.choice(eligible_actions)
