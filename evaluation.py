import json

from deepagents.backends import StateBackend
from langchain.chat_models import init_chat_model

from prompts import CORRECTION_INSTRUCTIONS

backend = StateBackend()


def generation_eval(user_query: str, llm_output: str):
    """Evaluate generation results loaded from the generation JSONL file."""
    print("Number 1", user_query)
    print("Number 2", llm_output)

    with open("eval/generation.jsonl", "r") as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        result = json.loads(json_str)
        grade_answer(user_query, result, llm_output)


def retrival_evaluation():
    """Evaluate retrieval results loaded from the retrieval JSONL file."""
    with open("eval/retrieval.jsonl", "r") as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        # print(f"Result: {result}")


def grade_answer(question: str, gold_answer: str, agent_answer: str) -> bool:
    """Generate a grading answer and return a bool if answer is living up to criteria"""
    queryString = createQuestion(question, gold_answer, agent_answer)
    response = init_chat_model("ollama:qwen2.5:14b").invoke(queryString)
    last_line = str(response.content).strip().splitlines()[-1]
    return last_line.startswith("GRADE: True")


def createQuestion(question: str, gold_answer: str, agent_answer: str):
    return f"""
    Instructions: {CORRECTION_INSTRUCTIONS}, 
    Question: {question}, 
    Gold answer: {gold_answer}, 
    Agent answer: {agent_answer}
    """


# Start med at få input og output herind.
# Still problem with passing in user_query / llm_output for every record in generation.jsonl, and then pass the whole result dict where grade_answer expects a gold_answer string.



# ----
# Lav dette om til en funktion
# Tag prompt input og output og kør det igennem funktionen.
# Returner bool hvis godkendt
