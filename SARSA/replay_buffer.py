# Stocker les expces
# quadruplets ... s(t), a(t), r(t), s(t+1)
# besoin du done (?)

import numpy as np


class ReplayBuffer:
    """classe de replay buffer"""
    def __init__(self):
        self.lst_expces = []
        self.batch_size = 50
    
    def add_expce(self, st, at, rt, st_next, done):
        """
        ajouter une expérience à chaque mvt fait dans le jeu
        """
        self.lst_expces.append((st, at, rt, st_next, done))
    
    def get_batch(self):
        """
        une fois de temps en temps, on récup un batch au pif pour MAJ la fonction Q
        """
        lst_st = []
        lst_at = []
        lst_rt = []
        lst_st_next = []
        lst_done = []

        for _ in range(self.batch_size):
            idx = np.random.randint(low=0, high=len(self.lst_expces))

            lst_st.append(self.lst_expces[idx][0])
            lst_at.append(self.lst_expces[idx][1])
            lst_rt.append(self.lst_expces[idx][2])
            lst_st_next.append(self.lst_expces[idx][3])
            lst_done.append(self.lst_expces[idx][4])
        
        return lst_st, lst_at, lst_rt, lst_st_next, lst_done
    
    def __len__(self):
        """
        retourne la taille de la liste. len peut être appelé juste en faisant "len(blabla)". Ca override la méthode len générale
        """
        return len(self.lst_expces)




