import os

from dotenv import load_dotenv
from mesa.agent import Agent
from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.model import Model
from mesa.space import (
    ContinuousSpace,
    MultiGrid,
    SingleGrid,
)

from mesa_llm import Plan
from mesa_llm.memory import Memory
from mesa_llm.module_llm import ModuleLLM
from mesa_llm.reasoning import (
    Observation,
    ReActReasoning,
    Reasoning,
    ReWOOReasoning,
)
from mesa_llm.tools.tool_manager import ToolManager


class LLMAgent(Agent):
    """
    LLMAgent manages an LLM backend and optionally connects to a memory module.

    Parameters:
        model (Model): The mesa model the agent in linked to.
        api_key (str): The API key for the LLM provider.
        llm_model (str): The model to use for the LLM in the format 'provider/model'. Defaults to 'openai/gpt-4o'.
        system_prompt (str | None): Optional system prompt to be used in LLM completions.
        reasoning (str): Optional reasoning method to be used in LLM completions.

    Attributes:
        llm (ModuleLLM): The internal LLM interface used by the agent.
        memory (Memory | None): The memory module attached to this agent, if any.

    """

    def __init__(
        self,
        model: Model,
        api_key: str,
        reasoning: type[Reasoning],
        llm_model: str = "openai/gpt-4o",
        system_prompt: str | None = None,
        vision: float | None = None,
        internal_state: list[str] | str | None = None,
    ):
        super().__init__(model=model)

        self.model = model

        self.llm = ModuleLLM(
            api_key=api_key, llm_model=llm_model, system_prompt=system_prompt
        )

        self.memory = Memory(
            agent=self,
            short_term_capacity=5,
            consolidation_capacity=2,
            api_key=api_key,
            llm_model=llm_model,
        )

        self.tool_manager = ToolManager()
        self.vision = vision
        self.reasoning = reasoning(agent=self)
        self.system_prompt = system_prompt

        if isinstance(internal_state, str):
            internal_state = [internal_state]
        elif internal_state is None:
            internal_state = []

        self.internal_state = internal_state

    def __str__(self):
        return f"LLMAgent {self.unique_id}"

    def apply_plan(self, plan: Plan) -> list[dict]:
        """
        Execute the plan in the simulation.
        """
        tool_call_resp = self.tool_manager.call_tools(
            agent=self, llm_response=plan.llm_plan
        )
        self.memory.add_to_memory(
            type="Tool_Call_Responses", content=str(tool_call_resp), step=plan.step
        )
        return tool_call_resp

    def generate_obs(self) -> Observation:
        """
        Returns an instance of the Observation dataclass enlisting everything the agent can see in the model in that step.

        If the agents vision is set to anything above 0, the agent will get the details of all agents falling in that radius.
        If the agents vision is set to -1, then the agent will get the details of all the agents present in the simulation at that step.
        If it is set to 0 or None, then no information is returned to the agent.

        """
        gen_obs = Observation()
        gen_obs.step = self.model.steps
        gen_obs.self_state = {
            "system_prompt": self.system_prompt,
            "location": self.pos,
            "internal_state": self.internal_state,
        }
        if self.vision is not None and self.vision > 0:
            if isinstance(
                self.model.grid,
                SingleGrid | MultiGrid | OrthogonalMooreGrid | OrthogonalVonNeumannGrid,
            ):
                neighbor_coords = self.model.grid.get_neighborhood(
                    self.pos, moore=True, include_center=False, radius=self.vision
                )
                # To get agents at those coordinates:
                neighbors = [self.model.grid[x, y] for x, y in neighbor_coords]
            elif isinstance(self.model.space, ContinuousSpace):
                neighbors, _ = self.get_neighbors_in_radius(radius=self.vision)
        elif self.vision == -1:
            all_agents = list(self.model.agents)
            neighbors = [agent for agent in all_agents if agent is not self]
        else:
            neighbors = []

        local_state = {}
        for i in neighbors:
            local_state[i.__class__.__name__ + " " + str(i.unique_id)] = {
                "position": i.pos,
                "internal_state": i.internal_state,
            }
        gen_obs.local_state = local_state

        return gen_obs

    def send_message(self, message: str, recipients: list[Agent]) -> str:
        """
        Send a message to the recipients.
        """
        for recipient in [*recipients, self]:
            recipient.memory.add_to_memory(
                type="Message",
                content=message,
                step=self.model.steps,
                metadata={
                    "sender": self,
                    "recipients": recipients,
                },
            )
        return f"{self} → {recipients} : {message}"


if __name__ == "__main__":
    model = Model()
    load_dotenv()

    llm_agent_colin = LLMAgent(
        model=model,
        api_key=os.getenv("GEMINI_API_KEY"),
        reasoning=ReActReasoning,
        llm_model="gemini/gemini-2.0-flash",
        system_prompt="You are an agent that is a part of a simulation. You are able to use tools to interact with the environment.",
    )

    llm_agent_sanika = LLMAgent(
        model=model,
        api_key=os.getenv("GEMINI_API_KEY"),
        reasoning=ReWOOReasoning,
        llm_model="gemini/gemini-2.0-flash",
        system_prompt="You are an agent that is a part of a simulation. You are able to use tools to interact with the environment.",
    )

    print(
        llm_agent_colin.send_message(
            "Isn't this working like a charm ?", [llm_agent_sanika]
        )
    )
    print(llm_agent_sanika.memory.format_short_term(), "\n\n")
    print(llm_agent_colin.memory.format_short_term(), "\n\n")
