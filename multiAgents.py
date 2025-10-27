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
    Un agente reflexivo que elige su acción en cada punto de decisión 
    evaluando heurísticamente los estados sucesores.
    """

    def getAction(self, gameState):
        """
        No es necesario modificar este método. Selecciona una acción legal 
        devolviendo aquella con mejor evaluación heurística.
        """
        # Lista de acciones legales disponibles
        legalMoves = gameState.getLegalActions()

        # Calcular puntajes para cada acción usando la función de evaluación
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        # Escoger índices de todas las acciones con puntaje máximo
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # Elegir aleatoriamente entre las mejores (para desempatar al azar)
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghost.scaredTimer for ghost in newGhostStates]
    
        # Estado terminal
        if successorGameState.isWin():
            return float('inf')
        if successorGameState.isLose():
            return -float('inf')
    
        score = successorGameState.getScore()
    
        # Comida: buscar la más cercana
        if newFood:
            minFoodDist = min(manhattanDistance(newPos, foodPos) for foodPos in newFood)
            score += 10.0 / minFoodDist
    
        # Fantasmas: evitar si no están asustados, acercarse si sí
        for ghost, scared in zip(newGhostStates, newScaredTimes):
            dist = manhattanDistance(newPos, ghost.getPosition())
            if scared == 0:
                if dist <= 1:
                    return -float('inf')
                score -= 2.0 / dist
            else:
                score += 2.0 / dist
    
        # Penalizar quedarse quieto
        if action == Directions.STOP:
            score -= 5
    
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
    Clase base abstracta para agentes adversarios multi-agente.
    Contiene la configuración de la función de evaluación y profundidad.
    """
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman es el agente 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """Agente que implementa búsqueda MiniMax adversaria."""
    
    def getAction(self, gameState):
        """Devuelve la mejor acción aplicando minimax hasta la profundidad self.depth."""
        # Función recursiva minimax: retorna el valor minimax del estado (para el agente actual).
        def minimax(agentIndex, currentDepth, gameState):
            # Si alcanzamos profundidad máxima o estado terminal, evaluar estado
            if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            # Calcular el siguiente agente (envolviendo al número total de agentes)
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            # Si volvemos al Pacman, incrementamos la profundidad (un ciclo completo de turnos)
            nextDepth = currentDepth + 1 if nextAgent == 0 else currentDepth

            # Obtener acciones legales (incluyendo STOP para generalidad)
            legalActions = gameState.getLegalActions(agentIndex)

            if agentIndex == 0:  # Turno de Pacman (MAX)
                maxVal = float("-inf")
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    val = minimax(nextAgent, nextDepth, successor)
                    maxVal = max(maxVal, val)
                return maxVal
            else:  # Turno de un fantasma (MIN)
                minVal = float("inf")
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    val = minimax(nextAgent, nextDepth, successor)
                    minVal = min(minVal, val)
                return minVal

        # Decidir la mejor acción mirando los valores minimax de los sucesores de Pacman
        bestAction = None
        bestValue = float("-inf")
        for action in gameState.getLegalActions(0):  # acciones de Pacman
            successor = gameState.generateSuccessor(0, action)
            value = minimax(1, 0, successor)  # evaluar minimax empezando con el primer fantasma
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """Agente que implementa minimax con poda Alpha-Beta."""
    
    def getAction(self, gameState):
        """Devuelve la mejor acción usando minimax con poda α-β hasta self.depth."""
        # Función recursiva con parámetros alpha y beta
        def alphabeta(agentIndex, currentDepth, gameState, alpha, beta):
            # Criterio terminal: profundidad máxima o estado gana/pierde
            if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = currentDepth + 1 if nextAgent == 0 else currentDepth
            legalActions = gameState.getLegalActions(agentIndex)

            if agentIndex == 0:  # Nodo Max (Pacman)
                value = float("-inf")
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = max(value, alphabeta(nextAgent, nextDepth, successor, alpha, beta))
                    alpha = max(alpha, value)
                    if value > beta:
                        # Corte (podar): este valor es mayor que el mínimo que el fantasma garantizaba
                        break
                return value
            else:  # Nodo Min (Fantasma)
                value = float("inf")
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = min(value, alphabeta(nextAgent, nextDepth, successor, alpha, beta))
                    beta = min(beta, value)
                    if value < alpha:
                        # Corte: este valor es menor que el máximo que Pacman ya garantizaba
                        break
                return value

        # Búsqueda inicial desde Pacman (agente 0) con α=-∞, β=+∞
        bestAction = None
        alpha = float("-inf")
        beta = float("inf")
        bestValue = float("-inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # Evaluar usando alpha-beta para el primer fantasma
            value = alphabeta(1, 0, successor, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            # Actualizar α en la raíz y continuar (no se poda a nivel raíz)
            alpha = max(alpha, bestValue)
        return bestAction



class ExpectimaxAgent(MultiAgentSearchAgent):
    """Agente que implementa búsqueda Expectimax (fantasmas aleatorios)."""
    
    def getAction(self, gameState):
        """Devuelve la acción óptima estimada mediante expectimax."""
        def expectimax(agentIndex, currentDepth, gameState):
            # Criterio terminal (igual que en minimax)
            if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = currentDepth + 1 if nextAgent == 0 else currentDepth
            legalActions = gameState.getLegalActions(agentIndex)

            if agentIndex == 0:  # Pacman (Maximizar)
                bestVal = float("-inf")
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    bestVal = max(bestVal, expectimax(nextAgent, nextDepth, successor))
                return bestVal
            else:  # Fantasma (Valor Esperado)
                totalVal = 0.0
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    totalVal += expectimax(nextAgent, nextDepth, successor)
                # Devolver valor promedio (asume distribución uniforme de acciones del fantasma)
                return totalVal / len(legalActions)

        # Elegir la acción con el mayor valor esperado
        bestAction = None
        bestValue = float("-inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(1, 0, successor)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction



def betterEvaluationFunction(currentGameState):
    """
    Una función de evaluación más sofisticada para Pacman.
    Retorna un valor numérico mayor para estados mejores para Pacman.
    """
    # Si el juego termina en este estado, devolver +-inf inmediatamente
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    pos = currentGameState.getPacmanPosition()      # posición de Pacman
    food = currentGameState.getFood().asList()      # lista de posiciones de comida restante
    ghostStates = currentGameState.getGhostStates() # estados de fantasmas
    capsules = currentGameState.getCapsules()       # posiciones de cápsulas restantes

    score = currentGameState.getScore()  # puntuación base del estado

    # Distancia a la comida más cercana (Manhattan)
    if food:
        minFoodDist = min(manhattanDistance(pos, foodPos) for foodPos in food)
        # Cuanto más cerca la comida, mayor el aporte (peso elegido empíricamente)
        score += 10.0 / minFoodDist
    else:
        # Si no queda comida, este es un estado ganador (caso manejado arriba como isWin)
        score += 10.0  # (este término realmente no importará si isWin ya retornó inf)

    # Considerar fantasmas: activo vs asustado
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        dist = manhattanDistance(pos, ghostPos)
        if dist == 0:
            # Pacman está en la misma casilla que un fantasma...
            if ghostState.scaredTimer == 0:
                return float("-inf")  # ...y el fantasma no está asustado (estado pésimo, casi derrota)
            # Si el fantasma está asustado y en la misma casilla, Pacman lo comería (puntos ya sumados en score)
            # No devolvemos inf aquí porque el juego continúa tras comer al fantasma
        else:
            if ghostState.scaredTimer > 0:
                # Fantasma asustado: incentiva acercarse (peso alto porque comer fantasmas vale mucho)
                score += 100.0 / dist
            else:
                # Fantasma activo: alejarse es mejor. Penalizar proximidad (peso negativo)
                score += -10.0 / dist

    # Penalizar ligeramente cápsulas restantes (menos cápsulas -> mejor estado)
    score += -4.0 * len(capsules)

    # (Opcionalmente se podría penalizar también la cantidad de comida restante, pero 
    # gran parte de eso ya está reflejado en score base y el término de distancia)

    return score

# Abreviatura para autograder (se suele hacer en el archivo original)
better = betterEvaluationFunction

better = betterEvaluationFunction

