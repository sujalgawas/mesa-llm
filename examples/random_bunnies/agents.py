from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_manager import ToolManager


class Bunny(LLMAgent):
    def __init__(
        self,
        model,
        api_key,
        reasoning,
        llm_model,
        vision,
        internal_state,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            reasoning=reasoning,
            llm_model=llm_model,
            vision=vision,
            internal_state=internal_state,
        )

        self.tool_manager = ToolManager()
        self.system_prompt = "You are a bunny that randomly moves around a grid."

    def step(self):
        prompt = "Move around the grid."
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=self.generate_obs(),
            selected_tools=["move_one_step", "teleport_to_location"],
        )
        self.apply_plan(plan)
        print(f"Bunny {self.unique_id} moved to {self.pos}")

    def generate_obs(self):
        return f"You are a bunny at position {self.pos}."
