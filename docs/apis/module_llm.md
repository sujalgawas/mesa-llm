# ModuleLLM

ModuleLLM provides a unified interface for integrating Large Language Models from multiple providers into Mesa-LLM agents. It abstracts away provider-specific implementation details while offering both synchronous and asynchronous generation capabilities with support for function calling, structured outputs, and automatic retry logic.

### class ModuleLLM(api_key, llm_model, system_prompt=None)

**Attributes:**
- **api_key** - API key for LLM provider
- **llm_model** - Model identifier in format 'provider/model'
- **system_prompt** - Optional system-level instructions for LLM context

**Parameters:**
- **api_key** (*str*) – API key for LLM provider
- **llm_model** (*str*) – Model identifier in format 'provider/model' (e.g., 'openai/gpt-4o')
- **system_prompt** (*str | None*) – Optional system prompt for LLM interactions

**Supported Providers:**
- OpenAI
- Anthropic
- xAI
- Huggingface
- OpenRouter
- NovitaAI
- Gemini
- Ollama

**Methods:**

**get_messages(prompt)** → *list[dict]*
Format prompt messages for LLM in expected message structure with role-based formatting.

**generate(prompt, tool_schema=None, tool_choice="auto", response_format=None)** → *str*
Generate LLM response with automatic retry logic and exponential backoff. Supports function calling and structured outputs.

**async agenerate(prompt, tool_schema=None, tool_choice="auto", response_format=None)** → *str*
Asynchronous version for parallel LLM calls in multi-agent simulations.

### Basic LLM Setup
In your .env file, set the API key for the LLM provider, then in your python file, call the ModuleLLM class with the desired model and system prompt.

```bash
# .env
OPENAI_API_KEY=your-api-key
ANTHROPIC_API_KEY=your-api-key
```


```python
# my_agent.py
from mesa_llm.module_llm import ModuleLLM

# Initialize with specific provider and model
llm = ModuleLLM(
    llm_model="openai/gpt-4o",
    system_prompt="You are a helpful simulation agent."
)

# Generate response
response = llm.generate("What should I do next in this situation?")
```

### Tool Integration

```python
from mesa_llm.tools.tool_manager import ToolManager

tool_manager = ToolManager()
llm = ModuleLLM(api_key="key", llm_model="openai/gpt-4o")

# Generate with tool calling
response = llm.generate(
    prompt="Move to a better location",
    tool_schema=tool_manager.get_all_tools_schema(),
    tool_choice="auto"
)

# Handle tool calls
if response.choices[0].message.tool_calls:
    tool_manager.call_tools(agent=agent, llm_response=response.choices[0].message)
```

### Asynchronous Usage

```python
async def generate_plan(llm, prompt, tools):
    """Generate response asynchronously for parallel processing"""
    response = await llm.agenerate(
        prompt=prompt,
        tool_schema=tools,
        tool_choice="required"
    )
    return response

# Use in parallel agent execution
responses = await asyncio.gather(*[
    generate_plan(agent.llm, prompt, tools) for agent in agents
])
```

### Structured Output

```python
from pydantic import BaseModel

class AgentDecision(BaseModel):
    reasoning: str
    action: str
    confidence: float

response = llm.generate(
    prompt="Analyze the situation and decide your action",
    response_format=AgentDecision
)

# Parse structured response
decision = AgentDecision.parse_raw(response.choices[0].message.content)
```

### Integration with LLMAgent

```python
class MyAgent(LLMAgent):
    def __init__(self, model, api_key, **kwargs):
        super().__init__(
            model=model,
            api_key=api_key,
            llm_model="anthropic/claude-3-sonnet",  # Automatically creates ModuleLLM
            system_prompt="Custom agent behavior instructions",
            **kwargs
        )

    def step(self):
        # Access LLM through self.llm
        response = self.llm.generate("What should I do?")
```