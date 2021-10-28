# import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class GridWorld:
    """crée une grille Gridworld"""

    def __init__(self):
        self.hauteur = 4
        self.largeur = 5
        self.emplacementAgent = (0, 0)
        self.obstacles = [(1, 2), (2, 1)]
        self.bombe = (self.hauteur - 2, self.largeur - 1)
        self.tresor = (self.hauteur - 1, self.largeur - 1)
        self.fin = [self.bombe, self.tresor]
        self.actions = ['haut', 'bas', 'gauche', 'droite']
        """crée la grille des récompenses"""
        self.grilleRecompenses = np.zeros((self.hauteur, self.largeur)) - 1
        self.grilleRecompenses[self.bombe[0], self.bombe[1]] = -10
        self.grilleRecompenses[self.tresor[0], self.tresor[1]] = 10

    def recompense(self, emplacement):
        """renvoie la récompense reçue par l'agent à cet emplacement"""
        return self.grilleRecompenses[emplacement[0], emplacement[1]]

    def avancer(self, action):
        """avance l'agent selon l'action placée en paramètre"""
        ici = self.emplacementAgent
        if action == 'haut':
            if ici[0] == 0 or (ici[0] - 1, ici[1]) in self.obstacles:
                recompense = self.recompense(ici)
            else:
                self.emplacementAgent = (ici[0] - 1, ici[1])
                recompense = self.recompense(ici)

        elif action == 'bas':
            if ici[0] == self.hauteur - 1 or (ici[0] + 1, ici[1]) in self.obstacles:
                recompense = self.recompense(ici)
            else:
                self.emplacementAgent = (ici[0] + 1, ici[1])
                recompense = self.recompense(ici)

        elif action == 'gauche':
            if ici[1] == 0 or (ici[0], ici[1] - 1) in self.obstacles:
                recompense = self.recompense(ici)
            else:
                self.emplacementAgent = (ici[0], ici[1] - 1)
                recompense = self.recompense(ici)

        elif action == 'droite':
            if ici[1] == self.largeur - 1 or (ici[0], ici[1] + 1) in self.obstacles:
                recompense = self.recompense(ici)
            else:
                self.emplacementAgent = (ici[0], ici[1] + 1)
                recompense = self.recompense(ici)

        return recompense


class RandomAgent():
    """agent qui choisit la direction à prendre aléatoirement"""

    def __init__(self, environnement):
        self.environnement = environnement

    def choixAction(self, actions):
        return np.random.choice(actions)


class Q_Agent():
    """agent qui agit selon l'algorithme du Q-learning"""

    def __init__(self, environnement, epsilon=0.05, alpha=0.1, gamma=1):
        self.environnement = environnement
        self.position = environnement.emplacementAgent
        """la Q-table sera composée pour chaque case de la grille d'un dictionnaire des 4 actions comme clés et des valeurs
        de Q(emplacement, action) comme valeur"""
        self.qTable = dict()
        for x in range(self.environnement.hauteur):
            for y in range(self.environnement.largeur):
                self.qTable[(x, y)] = {'haut': 0, 'bas': 0, 'gauche': 0, 'droite': 0}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def choixAction(self, actions):
        """choisit aléatoirement une action avec une proba de epsilon et sinon le max de la Q-table"""
        if np.random.uniform(0, 1) < self.epsilon:
            action = actions[np.random.randint(0, len(actions))]
        else:
            qValues = self.qTable[self.environnement.emplacementAgent]
            maxValue = max(qValues.values())
            choix = []
            for k, v in qValues.items():
                if v == maxValue:
                    choix.append(k)
            action = np.random.choice(choix)
        return action

    def learn(self, etatPrecedent, recompense, etatCourant, action):
        """met à jour la Q-table en fonction des états courant et précédent et de l'action choisie"""
        qValues = self.qTable[etatCourant]
        maxQValue = max(qValues.values())
        qValueCourante = self.qTable[etatPrecedent][action]
        """ Théorème de Bellmann appliqué, ici alpha reste constant"""
        self.qTable[etatPrecedent][action] = recompense + self.gamma * maxQValue
        # self.qTable[etatPrecedent][action] = (1 - self.alpha) * qValueCourante + self.alpha * (recompense + self.gamma * maxQValue)


def play(agent, essais=500, maxSteps=1000, discount=1, learn=False):
    """lance les simulations et renvoie une liste des récompenses obtenues pour chaque épisode"""
    recompenseParEpisode = []
    environnement = agent.environnement

    for essai in range(essais):
        cumulRecompense = 0
        step = 0
        GameOver = False
        while step < maxSteps and not GameOver:
            etatPrecedent = environnement.emplacementAgent
            action = agent.choixAction(environnement.actions)
            recompense = environnement.avancer(action)
            etatCourant = environnement.emplacementAgent

            if learn:
                agent.learn(etatPrecedent, recompense, etatCourant, action)

            cumulRecompense += discount * recompense
            step += 1

            if environnement.emplacementAgent in environnement.fin:
                """redémarre l'environnement à 0 pour un nouvel épisode"""
                environnement.__init__()
                GameOver = True

        recompenseParEpisode.append(cumulRecompense)

    return recompenseParEpisode


"""main : on lance d'abord un agent aléatoire"""
env = GridWorld()
agent = RandomAgent(env)
recompenseAgentAleat = play(agent, essais=500)
# plt.show()

"""on lance ensuite un Q-Agent"""

env = GridWorld()
agentQ = Q_Agent(env)
recompenseQAgent = play(agentQ, essais=500, maxSteps=1000, learn=True)

"""affichage des résultats"""
plt.plot(recompenseAgentAleat, label="agent aléatoire")
plt.plot(recompenseQAgent, label="agent entrainé")
plt.legend()
plt.title("Comparaison du score obtenu sur 500 essais entre un agent aléatoire et un au cours de son entraînement",
          fontsize=16)
# plt.show()

"""affichage des résultats du Q-agent"""
plt.plot(recompenseQAgent, label="agent entrainé", color="orange")
plt.title("Score obtenu sur 500 essais par un agent au cours de son entraînement", fontsize=16)
# plt.show()

"""affichage de la Q-table"""
for key, value in agentQ.qTable.items():
    print(str(key))
    print('\t' + str(value))

