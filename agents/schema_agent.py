import json
from langchain_core.messages import HumanMessage
from agent.llm import llm_router
from agent.llm_structured import parse_json_from_text
from agent.output_schemas import Selection
from agent.prompt_loader import render_prompt
from database.schema_inspector import inspect_database_schema, compact_schema
class SchemaSelectionAgent:
    def identify_collections(self, question, catalog=None):
        catalog = inspect_database_schema() if catalog is None else catalog
        collections = compact_schema(catalog)
        if not collections:
            return {"error": "no_collections", "detail": "The configured MongoDB database has no collections."}
        prompt = render_prompt("schema_selector", {"question": question, "schema_json": json.dumps(collections)})
        response = llm_router.invoke("schema_selection", [HumanMessage(content=prompt)], Selection)
        selected = Selection.model_validate(parse_json_from_text(response.content))
        if set(selected.fields) != set(selected.collections) or not set(selected.collections) <= set(collections):
            raise ValueError("Selection contains missing or unknown collections")
        result = {}
        for name in selected.collections:
            fields = collections[name]["fields"]
            requested = selected.fields[name]
            if not set(requested) <= set(fields):
                raise ValueError("Selection contains unknown fields")
            result[name] = {"fields": {key: fields[key] for key in requested}}
        return {"collections": result}
SchemaAgent = SchemaSelectionAgent
