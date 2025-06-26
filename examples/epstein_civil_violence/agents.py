import math
from enum import Enum

import mesa

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_manager import ToolManager

citizen_tool_manager = ToolManager()
cop_tool_manager = ToolManager()


class CitizenState(Enum):
    QUIET = 1
    ACTIVE = 2
    ARRESTED = 3


class Citizen(LLMAgent, mesa.discrete_space.CellAgent):
    """
    A member of the general population, may or may not be in active rebellion.
    Summary of rule: If grievance - risk > threshold, rebel.

    Attributes:
        hardship: Agent's 'perceived hardship (i.e., physical or economic
            privation).' Exogenous, drawn from U(0,1).
        regime_legitimacy: Agent's perception of regime legitimacy, equal
            across agents.  Exogenous.
        risk_aversion: Exogenous, drawn from U(0,1).
        threshold: if (grievance - (risk_aversion * arrest_probability)) >
            threshold, go/remain Active
        vision: number of cells in each direction (N, S, E and W) that agent
            can inspect
        condition: Can be "Quiescent" or "Active;" deterministic function of
            greivance, perceived risk, and
        grievance: deterministic function of hardship and regime_legitimacy;
            how aggrieved is agent at the regime?
        arrest_probability: agent's assessment of arrest probability, given
            rebellion
    """

    def __init__(
        self,
        model,
        api_key,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        regime_legitimacy=0.5,
        arrest_prob_constant=0.5,
        threshold=0.5,
    ):
        # Call the superclass constructor with updated internal state
        super().__init__(
            model=model,
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )

        self.hardship = self.random.random()
        self.risk_aversion = self.random.random()
        self.regime_legitimacy = regime_legitimacy
        self.threshold = threshold
        self.state = CitizenState.QUIET
        self.vision = vision
        self.jail_sentence_left = 0  # A jail sentence of 1 implies that the agent cannot participate in the next 10 steps.
        self.grievance = self.hardship * (1 - self.regime_legitimacy)
        self.arrest_prob_constant = arrest_prob_constant
        self.arrest_probability = None

        self.neighborhood = []
        self.neighbors = []
        self.empty_neighbors = []

        self.tool_manager = citizen_tool_manager
        self.system_prompt = "You are a citizen in a country that is experiencing civil violence. You are a member of the general population, may or may not be in active rebellion. You can move one step in a nearby cell or change your state."

    def step(self):
        if self.jail_sentence_left == 0:
            observation = self.generate_obs()
            prompt = "Move around and change your state if necessary."
            plan = self.reasoning.plan(
                prompt=prompt,
                obs=observation,
                selected_tools=["change_state", "move_citizen"],
            )
            self.apply_plan(plan)
        else:
            self.jail_sentence_left -= 0.1

    def update_estimated_arrest_probability(self):
        """
        Based on the ratio of cops to actives in my neighborhood, estimate the
        p(Arrest | I go active).
        """
        cops_in_vision = 0
        actives_in_vision = 1  # citizen counts herself
        for neighbor in self.neighbors:
            if isinstance(neighbor, Cop):
                cops_in_vision += 1
            elif neighbor.state == CitizenState.ACTIVE:
                actives_in_vision += 1

        # there is a body of literature on this equation
        # the round is not in the pnas paper but without it, its impossible to replicate
        # the dynamics shown there.
        self.arrest_probability = 1 - math.exp(
            -1 * self.arrest_prob_constant * round(cops_in_vision / actives_in_vision)
        )


class Cop(LLMAgent, mesa.discrete_space.CellAgent):
    """
    A cop for life.  No defection.
    Summary of rule: Inspect local vision and arrest a random active agent.

    Attributes:
        unique_id: unique int
        x, y: Grid coordinates
        vision: number of cells in each direction (N, S, E and W) that cop is
            able to inspect
    """

    def __init__(
        self,
        model,
        api_key,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        max_jail_term=2,
    ):
        """
        Create a new Cop.
        Args:
            x, y: Grid coordinates
            vision: number of cells in each direction (N, S, E and W) that
                agent can inspect. Exogenous.
            model: model instance
        """
        super().__init__(
            model=model,
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )
        self.max_jail_term = max_jail_term
        self.tool_manager = cop_tool_manager

    def step(self):
        """
        Inspect local vision and arrest a random active agent. Move if
        applicable.
        """
        observation = self.generate_obs()
        prompt = "Inspect your local vision and arrest a random active agent. Move if applicable."
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["move_cop", "arrest_citizen"],
        )
        self.apply_plan(plan)
