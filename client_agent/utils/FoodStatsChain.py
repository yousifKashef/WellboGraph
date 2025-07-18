from typing import Dict, List


from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class Portion(BaseModel):
    description: str = Field(..., description="The description of this portion")
    quantity: float = Field(..., description="Numerical Quantity")
    unit: str = Field(..., description="Unit of the serving (e.g., 'gram', 'liter')")


class FoodDetails(BaseModel):
    food_name: str = Field(..., description="Name of the food")

    nutrients: Dict[str, float] = Field(
        ...,
        description="Nutrients; keys are nutrient names, values are amounts."
    )

    portion: Portion = Field(
        ...,
        description=(
            "Portion info; like sandwich, bowl, tablespoon… also includes its "
            "numerical amount and unit of measurement"
        )
    )

    ingredients: List[str] = Field(
        ...,
        description="List of ingredient names."
    )

    recipe: str = Field(
        ...,
        description="Very brief overview of preparation."
    )


class FoodStatsChain:
    model = ChatOpenAI(model_name="gpt-4o")
    output_parser = PydanticOutputParser(pydantic_object=FoodDetails)

    prompt = ChatPromptTemplate.from_template(
        """
        You are a highly accurate nutrition bot that knows all foods worldwide.

        Provide detailed nutritional information for "{food_name}" including:

        - **nutrients** (exact values in grams or mg):
          Calories, Protein, Carbohydrates, Total Fat, Saturated Fat, Monounsaturated Fat,
          Polyunsaturated Fat, Omega-3 Fatty Acids, Omega-6 Fatty Acids, Fiber,
          Sugars, Cholesterol (mg), Sodium (mg), Potassium (mg), Calcium (mg),
          Iron (mg), Magnesium (mg), Zinc (mg), Vitamin A (IU or µg), 
          Vitamin C (mg), Vitamin D (IU or µg).

        - **Portion**:
          Provide portion name and info. If it is not provided, assign a common portion
          name for this food. If it’s 1 sandwich, say that in the description. For
          quantity put its weight in grams and for unit put "grams".

        - **ingredients** (list main ingredients).

        - **recipe** (short 2-3 sentences).

        Strictly follow schema:
        {format_instructions}

        Food: "{food_name}"
        """
    ).partial(format_instructions=output_parser.get_format_instructions())

    chain = prompt | model | output_parser
