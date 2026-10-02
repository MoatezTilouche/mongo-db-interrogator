from langchain_core.messages import HumanMessage
from agent.llm import llm_router
from agent.llm_structured import parse_json_from_text
from agent.output_schemas import Route
from agent.prompt_loader import render_prompt
class QueryRouter:
    """Deprecated compatibility wrapper; QueryContextAgent owns active routing."""
    def route(self, question):
        response = llm_router.invoke("query_router", [HumanMessage(content=render_prompt("query_router", {"question": question}))], Route)
        return Route.model_validate(parse_json_from_text(response.content)).model_dump()
