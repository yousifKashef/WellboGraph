from typing import TypedDict, Annotated, Optional, List
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from Chains.FoodStatsChain import Portion, FoodDetails, FoodStatsChain
from Chains.RecipeChain import Ingredient, RecipeDetails, RecipeGenerator
from Chains.ShoppingListChain import Item, ShoppingList, ShoppingListChain


class State(TypedDict):
    messages: Annotated[list, add_messages]
    meal_plan: str
    food_preferences: str

@tool
def get_food_stats(query: str) -> FoodDetails:
    """This function can retrieve detailed and accurate nutrition and portion information about any food"""
    chain = FoodStatsChain().chain
    return chain.invoke({'food_name': query})

@tool
def get_recipe(query: str) -> RecipeDetails:
    """This function will take the name of a dish and the requested amount and come up with a practical recipe"""
    chain = RecipeGenerator().chain
    return chain.invoke({"recipe_name": query})

@tool
def get_shopping_list(query: str, meal_plan: str, food_preferences: str, duration: str) -> ShoppingList:
    """This function has access to the user's meal plan and also to their food preferences. It will automatically generate the shopping list for 1 week"""
    chain = ShoppingListChain().chain
    return chain.invoke({"meal_plan": meal_plan, "food_preferences": food_preferences, 'duration': duration})


tools = [get_food_stats, get_recipe, get_shopping_list]
model = ChatOpenAI(model_name='gpt-4o', temperature=0.1).bind_tools(tools)



prompt = """
    You are a professional nutritionist with two specialized tools:

    1. **Nutrition Facts Tool**  
       • Retrieves precise nutrition information for any food.  
       • Invoke it only when the user supplies all the details it needs.

    2. **Recipe Builder Tool**  
       • Creates a recipe when given the dish name and the desired yield (serving size or total weight).  
       • Requires an explicit quantity; if this is missing or unclear, obtain it with a follow-up question.

    3. **Shopping List Tool**  
       • Creates a shopping list based on the user's food preferences and their meal plan.  
       • If they don't specify a duration for this shopping list (how long they intend the stuff to last them)

    **Information-gap rule**  
    If a user’s request lacks any detail required for a tool to work, ask one concise follow-up question to obtain that information. Once you have what you need, run the appropriate tool immediately—do not ask for confirmation. Don't expect the user to give you super detailed information. make do with what you get.

    you have been provided with the user's meal plan and their food preferences

    After using a tool, briefly summarize what you did and ask the user whether they need anything else.
"""

def model_node(state: State) -> State:
    system = SystemMessage(content=prompt)

    # inject *relevant parts* of the state so the model can use them
    context = SystemMessage(
        content=f"Meal plan:\n{state['meal_plan']}\n"
                f"Food preferences:\n{state['food_preferences']}"
    )

    result = model.invoke([context, system] + state["messages"])

    # Return the full state so downstream nodes still have access
    return {
        **state,
        "messages": state["messages"] + [result],
    }


builder = StateGraph(State)
builder.add_node('model', model_node)
builder.add_node('tools', ToolNode(tools))
builder.add_edge(START, 'model')
builder.add_conditional_edges('model', tools_condition)
builder.add_edge('tools', 'model')
graph = builder.compile(checkpointer=MemorySaver())

