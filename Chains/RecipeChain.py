from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from typing import Dict, List
from pydantic import BaseModel, Field

class Ingredient(BaseModel):
    name: str = Field(..., description="Name of the ingredient")
    quantity: float = Field(..., description="Numerical quantity")
    unit: str = Field(..., description="Measurement unit, e.g., 'gram', 'cup', 'teaspoon'")

class RecipeDetails(BaseModel):
    recipe_name: str = Field(..., description="Name of the recipe")

    ingredients: List[Ingredient] = Field(
        ...,
        description="List of ingredients with quantity and measurement units"
    )

    steps: List[str] = Field(
        ...,
        description="Ordered list of steps describing preparation"
    )

    serving_size: int = Field(
        ...,
        description="Number of servings the recipe yields"
    )

    nutrients: Dict[str, float] = Field(
        ...,
        description="Nutrition information per serving; keys are nutrient names, values are amounts."
    )


class RecipeGenerator:
    model = ChatOpenAI(model_name='gpt-4o')

    output_parser = PydanticOutputParser(pydantic_object=RecipeDetails)

    prompt = ChatPromptTemplate.from_template("""
    You are an expert culinary assistant knowledgeable about recipes from all cuisines.

    Provide a detailed recipe for "{recipe_name}", strictly adhering to this schema:

    - **Ingredients**:
      List each ingredient with its exact quantity and measurement unit (e.g., grams, cups, tablespoons).

    - **Steps**:
      Provide a clear, numbered sequence of steps to prepare this recipe.

    - **Serving Size**:
      State how many servings this recipe yields.

    - **Nutrients** (per serving):
      Provide exact nutritional values including Calories, Protein, Carbohydrates, Total Fat, Fiber, Sugars, Cholesterol (mg), Sodium (mg), Calcium (mg), Iron (mg), Vitamin C (mg).

    Strictly follow schema:
    {format_instructions}

    Recipe: "{recipe_name}"
    """)

    prompt = prompt.partial(format_instructions=output_parser.get_format_instructions())

    chain = prompt | model | output_parser

