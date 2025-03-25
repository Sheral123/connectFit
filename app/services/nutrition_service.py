from langchain import LLMChain, PromptTemplate
from app.services.ai_utils import GeminiLLM

def get_nutrition_advice(query: str) -> str:
    llm = GeminiLLM()
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template="User Query: {query}\n\nProvide clear, concise, and evidence-based nutritional advice."
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(query=query)
