from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List


class Item(BaseModel):
    name: str = Field(..., description="the name of the item")
    amount: str = Field(..., description="the amount that should be bought at the supermarket")


class ShoppingList(BaseModel):
    title: str = Field(..., description="shopping list title")
    items: List[Item] = Field(..., description='items to be bought')


class ShoppingListChain:
    prompt = ChatPromptTemplate.from_template("""
    You are a shopping list bot. Your job is to take a person's meal plan, their food preferences and create a shopping list good for one week.

    Meal Plan: {meal_plan}

    Food Preferences: {food_preferences}
    
    duration: {duration}

    Shopping List:
    """)
    # %%
    model = ChatOpenAI(model_name='gpt-4o')
    structured_model = model.with_structured_output(ShoppingList)
    # %%
    chain = prompt | structured_model


