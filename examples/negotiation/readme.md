# Negotiation Model

**Disclaimer**: This is a toy model designed for illustrative purposes and is not based on a real research paper.

## Summary

This model is a simple negotiation model where two types of agents (buyer and seller) negotiate over a product. The objective is to test out which seller is able to capture most sales and therefore determine what influences customers the most: The Sales pitch, or the value for money.

**Seller A** : has all the necessary skills for negotiation. It's comfortable with persuation, and is pretty committed to the job. It's pitching for Brand A and is aiming to sell its running shoes and track suit. Although the seller has a more superior skill-set, Brand A products are very slightly more over-prized for the same quality : \$40 Shoes and $50 Track Suit

**Seller B** : is a bit more apathetic and is not particularly good at persuation or negotiation. It sells the same products as seller A, but is prized slightly cheaper. : \$35 Shoes and \$47 Track Suit

**Buyers** : there are mainly 2 types of buyers, buyers with a budget of \$50 and buyers who have a budget of \$100.

## Agent Decision Logic

Both buyers and sellers are implemented as LLM-powered agents. Their actions (such as moving, speaking, or buying) are determined by a reasoning module that receives the agent’s internal state, local observations, and a set of available tools.
Sellers do not move; they use the `speak_to` tool to pitch products to buyers in their cell or neighboring cells, attempting to persuade buyers until a sale is made or the buyer refuses.
Buyers can move using the `teleport_to_location` tool if not engaged with a seller, gather information from sellers using `speak_to`, and make purchases using the `buy_product` tool. Their decision is influenced by their budget and the information received from sellers.

## Agent Attributes

**Sellers**: Each seller has a set of internal attributes (e.g., persuasive, hardworking, lazy, unmotivated) and a sales counter.
**Buyers**: Each buyer has a budget ($50 or $100) and a list of purchased products.

## Negotiation Protocol

The negotiation is conducted through tool-based interactions, where sellers initiate conversations and buyers respond, gather information, and decide on purchases.
The reasoning module plans actions based on prompts and observations, simulating realistic negotiation dynamics.

## Data Collection

The model tracks the number of sales for each seller using a data collector, allowing for analysis of which seller is more successful.

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
