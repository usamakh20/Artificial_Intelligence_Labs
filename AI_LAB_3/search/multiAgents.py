# multiAgents.py
#By: Talha Saeed Khan (130956), Usama Khalid (134476)  BSCS-5B

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        return successorGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        PACMAN = 0
        def max_agent(state, depth):
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(PACMAN)
            best_score = float("-inf")
            score = best_score
            best_action = Directions.STOP
            for action in actions:
                score = exp_agent(state.generateSuccessor(PACMAN, action), depth, 1)
                if score > best_score:
                    best_score = score
                    best_action = action
            if depth == 0:
                return best_action
            else:
                return best_score

        def exp_agent(state, depth, ghost):
            if state.isLose() or state.isWin():
                return state.getScore()
            next_ghost = ghost + 1
            if ghost == state.getNumAgents() - 1:
                # Although I call this variable next_ghost, at this point we are referring to a pacman agent.
                # I never changed the variable name and now I feel bad. That's why I am writing this guilty comment :(
                next_ghost = PACMAN
            actions = state.getLegalActions(ghost)
            best_score = float("inf")
            score = best_score
            for action in actions:
                if next_ghost == PACMAN: # We are on the last ghost and it will be Pacman's turn next.
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                    else:
                        score = max_agent(state.generateSuccessor(ghost, action), depth + 1)
                else:
                    score = exp_agent(state.generateSuccessor(ghost, action), depth, next_ghost)
                if score < best_score:
                    best_score = score
            return best_score
        return max_agent(gameState, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        PACMAN = 0
        def max_agent(state, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(PACMAN)
            best_score = float("-inf")
            score = best_score
            best_action = Directions.STOP
            for action in actions:
                score = min_agent(state.generateSuccessor(PACMAN, action), depth, 1, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score)
                if best_score > beta:
                    return best_score
            if depth == 0:
                return best_action
            else:
                return best_score

        def min_agent(state, depth, ghost, alpha, beta):
            if state.isLose() or state.isWin():
                return state.getScore()
            next_ghost = ghost + 1
            if ghost == state.getNumAgents() - 1:
                # Although I call this variable next_ghost, at this point we are referring to a pacman agent.
                # I never changed the variable name and now I feel bad. That's why I am writing this guilty comment :(
                next_ghost = PACMAN
            actions = state.getLegalActions(ghost)
            best_score = float("inf")
            score = best_score
            for action in actions:
                if next_ghost == PACMAN: # We are on the last ghost and it will be Pacman's turn next.
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                    else:
                        score = max_agent(state.generateSuccessor(ghost, action), depth + 1, alpha, beta)
                else:
                    score = min_agent(state.generateSuccessor(ghost, action), depth, next_ghost, alpha, beta)
                if score < best_score:
                    best_score = score
                beta = min(beta, best_score)
                if best_score < alpha:
                    return best_score
            return best_score
        return max_agent(gameState, 0, float("-inf"), float("inf"))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
Returns the expectimax action using self.depth and self.evaluationFunction
All ghosts should be modeled as choosing uniformly at random from their
legal moves.
"""
        "*** YOUR CODE HERE ***"
        def expectedvalue(gameState, agentindex, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            numghosts = gameState.getNumAgents() - 1
            legalActions = gameState.getLegalActions(agentindex)
            numactions = len(legalActions)
            totalvalue = 0
            for action in legalActions:
                nextState = gameState.generateSuccessor(agentindex, action)
                if (agentindex == numghosts):
                    totalvalue += maxvalue(nextState, depth - 1)
                else:
                    totalvalue += expectedvalue(nextState, agentindex + 1, depth)
            return totalvalue / numactions
        def maxvalue(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(0)
            bestAction = Directions.STOP
            score = -(float("inf"))
            for action in legalActions:
                prevscore = score
                nextState = gameState.generateSuccessor(0, action)
                score = max(score, expectedvalue(nextState, 1, depth))
            return score
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalActions = gameState.getLegalActions(0)
        bestaction = Directions.STOP
        score = -(float("inf"))
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            prevscore = score
            score = max(score, expectedvalue(nextState, 1, self.depth))
            if score > prevscore:
                bestaction = action
        return bestaction
        
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

