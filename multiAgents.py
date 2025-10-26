# multiAgents.py
# --------------
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
from game import Agent
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #def evaluationFunction(self, currentGameState, action):
        """
        Evalúa (estado, acción) mirando el estado sucesor.
        Más alto = mejor para Pacman.
        """
        from util import manhattanDistance
        from game import Directions
    
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFoodGrid = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        capsules = successorGameState.getCapsules()
    
        score = successorGameState.getScore()
    
        # Penaliza parar
        if action == Directions.STOP:
            score -= 10.0
    
        # Comida: bonus por comer y por acercarse a la más cercana
        oldFoodCount = currentGameState.getNumFood()
        newFoodCount = successorGameState.getNumFood()
        if newFoodCount < oldFoodCount:
            score += 20.0
    
        foodList = newFoodGrid.asList()
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, f) for f in foodList)
            score += 5.0 / (1.0 + minFoodDist)
    
        # Cápsulas: comer o acercarse
        if newPos in capsules:
            score += 25.0
        elif capsules:
            minCapDist = min(manhattanDistance(newPos, c) for c in capsules)
            score += 2.5 / (1.0 + minCapDist)
    
        # Fantasmas: evitar si no están asustados, perseguir si sí
        for gState, scared in zip(newGhostStates, newScaredTimes):
            gpos = gState.getPosition()
            d = manhattanDistance(newPos, gpos)
            if d == 0 and scared == 0:
                score -= 500.0
            elif scared > 0:
                score += 30.0 / (1.0 + d)
            else:
                if d <= 1:
                    score -= 300.0
                else:
                    score -= 6.0 / (1.0 + d)
    
        return score



def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
    """Devuelve la acción minimax usando self.depth y self.evaluationFunction."""
    def value(state, depth, agentIndex):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if agentIndex == 0:
            return maxValue(state, depth)
        else:
            return minValue(state, depth, agentIndex)

    def maxValue(state, depth):
        v = -float("inf")
        for a in state.getLegalActions(0):
            succ = state.generateSuccessor(0, a)
            v = max(v, value(succ, depth, 1 if state.getNumAgents() > 1 else 0))
        return v

    def minValue(state, depth, agentIndex):
        v = float("inf")
        numAgents = state.getNumAgents()
        nextAgent = agentIndex + 1
        nextDepth = depth
        if nextAgent == numAgents:
            nextAgent = 0
            nextDepth = depth + 1
        for a in state.getLegalActions(agentIndex):
            succ = state.generateSuccessor(agentIndex, a)
            v = min(v, value(succ, nextDepth, nextAgent))
        return v

    bestScore, bestAction = -float("inf"), None
    for a in gameState.getLegalActions(0):
        succ = gameState.generateSuccessor(0, a)
        score = value(succ, 0, 1 if gameState.getNumAgents() > 1 else 0)
        if score > bestScore:
            bestScore, bestAction = score, a
    return bestAction


    def getAction(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Agente Minimax con poda alfa-beta (Pregunta 3).
    Soporta múltiples fantasmas (múltiples capas de MIN).
    """

    def getAction(self, game_state):
        """
        Devuelve la acción minimax usando self.depth y self.evaluationFunction,
        aplicando poda alfa-beta. No reordena hijos.
        """
        def value(state, depth, agent_index, alpha, beta):
            # Criterios de corte
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # Pacman = MAX (agente 0), fantasmas = MIN (agentes >= 1)
            if agent_index == 0:
                return max_value(state, depth, alpha, beta)
            else:
                return min_value(state, depth, agent_index, alpha, beta)

        def max_value(state, depth, alpha, beta):
            v = -float("inf")
            # Pacman (agente 0)
            for a in state.getLegalActions(0):
                succ = state.generateSuccessor(0, a)
                # Siguiente agente: 1 (si hay fantasmas)
                v = max(v, value(succ, depth, 1 if state.getNumAgents() > 1 else 0, alpha, beta))
                # Poda
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, depth, agent_index, alpha, beta):
            v = float("inf")
            num_agents = state.getNumAgents()

            # Calcular siguiente (agente, profundidad)
            next_agent = agent_index + 1
            next_depth = depth
            if next_agent == num_agents:  # vuelta a Pacman
                next_agent = 0
                next_depth = depth + 1

            for a in state.getLegalActions(agent_index):
                succ = state.generateSuccessor(agent_index, a)
                v = min(v, value(succ, next_depth, next_agent, alpha, beta))
                # Poda
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        # Raíz: elegimos la mejor acción para Pacman
        alpha, beta = -float("inf"), float("inf")
        best_score, best_action = -float("inf"), None

        for a in game_state.getLegalActions(0):
            succ = game_state.generateSuccessor(0, a)
            score = value(succ, 0, 1 if game_state.getNumAgents() > 1 else 0, alpha, beta)
            if score > best_score:
                best_score, best_action = score, a
            # Actualiza alfa y posible poda en la raíz
            if best_score > beta:
                return best_action
            alpha = max(alpha, best_score)

        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """Expectimax: los fantasmas eligen uniformemente al azar entre sus acciones legales."""
        def value(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return maxValue(state, depth)
            else:
                return expValue(state, depth, agentIndex)
    
        def maxValue(state, depth):
            v = -float("inf")
            for a in state.getLegalActions(0):
                succ = state.generateSuccessor(0, a)
                v = max(v, value(succ, depth, 1 if state.getNumAgents() > 1 else 0))
            return v
    
        def expValue(state, depth, agentIndex):
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            numAgents = state.getNumAgents()
            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth = depth + 1
            p = 1.0 / len(actions)
            ex = 0.0
            for a in actions:
                succ = state.generateSuccessor(agentIndex, a)
                ex += p * value(succ, nextDepth, nextAgent)
            return ex
    
        bestScore, bestAction = -float("inf"), None
        for a in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, a)
            score = value(succ, 0, 1 if gameState.getNumAgents() > 1 else 0)
            if score > bestScore:
                bestScore, bestAction = score, a
        return bestAction



def betterEvaluationFunction(currentGameState):
    """
    Evaluación avanzada para Pacman (Pregunta 5).
    
    Combina múltiples factores:
      - Puntuación base del estado actual.
      - Distancia a la comida más cercana (entre más cerca, mejor).
      - Número total de comidas restantes (menos = mejor).
      - Distancia a las cápsulas de poder (entre más cerca, mejor).
      - Distancia a los fantasmas:
          • Si están asustados, se premia acercarse.
          • Si no están asustados, se penaliza acercarse.
      - Bonificaciones por ganar y penalizaciones por perder.
    """

    from util import manhattanDistance

    # --- Estados terminales ---
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return -float('inf')

    # --- Información del estado ---
    pacman_pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghost_states = currentGameState.getGhostStates()

    # Puntuación base del juego
    score = currentGameState.getScore()

    # --- Comida ---
    num_food = len(food)
    min_food_dist = min((manhattanDistance(pacman_pos, f) for f in food), default=1)

    # --- Cápsulas ---
    num_caps = len(capsules)
    min_cap_dist = min((manhattanDistance(pacman_pos, c) for c in capsules), default=1)

    # --- Fantasmas ---
    repel = 0.0   # penalización por acercarse a fantasmas activos
    attract = 0.0 # bonificación por acercarse a fantasmas asustados

    for ghost in ghost_states:
        dist = manhattanDistance(pacman_pos, ghost.getPosition())
        scared = ghost.scaredTimer

        if dist == 0 and scared == 0:
            # Pacman muere: estado terrible
            return -float('inf')
        elif scared > 0:
            attract += 2.0 / dist  # incentivo a perseguir fantasmas asustados
        else:
            repel += 4.0 / dist    # castigo a acercarse a fantasmas activos

    # --- Heurística combinada ---
    heuristic = 0.0
    heuristic += -2.5 * num_food
    heuristic += -10.0 * num_caps
    heuristic += 3.0 / min_food_dist
    heuristic += 2.0 / min_cap_dist
    heuristic += -repel + attract

    return score + heuristic


# Abbreviation
better = betterEvaluationFunction

