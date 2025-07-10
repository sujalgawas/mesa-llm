# Epstein Civil Violence Model

## Summary

This model is based on Joshua Epstein's simulation of how civil unrest grows and is suppressed. Citizen agents wander the grid randomly, and are endowed with individual risk aversion and hardship levels; there is also a universal regime legitimacy value. There are also Cop agents, who work on behalf of the regime. Cops arrest Citizens who are actively rebelling; Citizens decide whether to rebel based on their hardship and the regime legitimacy, and their perceived probability of arrest.

The model generates mass uprising as self-reinforcing processes: if enough agents are rebelling, the probability of any individual agent being arrested is reduced, making more agents more likely to join the uprising. However, the more rebelling Citizens the Cops arrest, the less likely additional agents become to join.

This model is implemented using Mesa-LLM unlike the original Mesa and NetLogo versions. All agents have the ability to think and use tools (arresting citizens for cops, changing their state, etc.) depending on their reasoning method.

## Technical Details

The **Epstein Civil Violence Model** simulates the dynamics of civil unrest using two types of agents: **Citizens** and **Cops**.

Each **Citizen** agent is characterized by individual attributes such as:

- **Hardship**
- **Risk aversion**
- **Threshold for rebellion**

Additionally, all Citizens share a common perception of **regime legitimacy**.

---

### Rebellion Rule

A Citizen becomes (or remains) *active* (i.e., rebels) if the following condition is met:

\[
\text{grievance} - (\text{risk\_aversion} \times \text{arrest\_probability}) > \text{threshold}
\]

Where:

\[
\text{grievance} = \text{hardship} \times (1 - \text{regime\_legitimacy})
\]

---

### Arrest Probability

The perceived probability of arrest is calculated as:

\[
\text{arrest\_probability} = 1 - \exp\left(-k \times \text{round}\left(\frac{\text{cops\_in\_vision}}{\text{actives\_in\_vision}}\right)\right)
\]

Where:

- \( k \) is a constant
- \( \text{cops\_in\_vision} \) is the number of cops within the agent’s vision
- \( \text{actives\_in\_vision} \) is the number of active Citizens (including the agent itself)

---

### Cops' Behavior

**Cops** patrol the grid and attempt to **arrest active Citizens** within their vision.

---

### LLM-Powered Agents

Both Citizens and Cops are implemented as **LLM-powered agents**, meaning:

- Their actions (e.g., `move`, `change state`, `arrest`) are determined by a **reasoning module**.
- This module takes as input:
  - The agent’s internal state
  - Local observations
  - A set of available tools

This design enables **flexible, context-aware decision-making** that incorporates both:

- Quantitative attributes (e.g., hardship, risk)
- Qualitative reasoning (e.g., situational awareness, adaptive strategy)


## How to Run

To run the model interactively, in this directory, run the following command

```
    $ solara run app.py
```

## Files

* ``model.py``: Core model code.
* ``agent.py``: Agent classes.
* ``app.py``: Sets up the interactive visualization.
* ``tools.py``: Tools for the llm-agents to use.

## Further Reading

This model is based adapted from:

[Epstein, J. “Modeling civil violence: An agent-based computational approach”, Proceedings of the National Academy of Sciences, Vol. 99, Suppl. 3, May 14, 2002](http://www.pnas.org/content/99/suppl.3/7243.short)

A similar model is also included with NetLogo:

Wilensky, U. (2004). NetLogo Rebellion model. http://ccl.northwestern.edu/netlogo/models/Rebellion. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

You can also find Mesa's version of the model without using LLMs here:
https://github.com/projectmesa/mesa/tree/main/mesa/examples/advanced/epstein_civil_violence