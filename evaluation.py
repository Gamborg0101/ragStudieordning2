import json
import time

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from rapidfuzz import fuzz

from datacollection.vector_store import vector_store
from index import agent
from prompts import CORRECTION_INSTRUCTIONS

AGENT_INVOKE_ATTEMPTS = 3


def invoke_agent_with_retry(question: str, attempts: int = AGENT_INVOKE_ATTEMPTS):
    """Retry agent.invoke on transient Ollama stream failures (see langchain-ai/langchain#34918)."""
    for attempt in range(1, attempts + 1):
        try:
            return agent.invoke({"messages": [HumanMessage(content=question)]})
        except ValueError as e:
            if (
                "No data received from Ollama stream" not in str(e)
                or attempt == attempts
            ):
                raise
            print(
                f"Ollama stream failed (attempt {attempt}/{attempts}), retrying: {question!r}"
            )
            time.sleep(2)


def generation_eval():
    """Evaluate generation results loaded from the generation JSONL file to evaulate output quality."""
    with open("eval/generation.jsonl", "r") as json_file:
        json_list = list(json_file)
        generation_results = []
    for json_str in json_list:
        record = json.loads(json_str)
        result = invoke_agent_with_retry(record["question"])
        llm_output = [message.text for message in result["messages"] if message.text]
        # print("LLM_OUPUT: ", llm_output)
        grade_answered = grade_answer(
            record["question"], record["gold_answer"], llm_output
        )
        generation_results.append(
            {
                "id": record["id"],
                "grade": grade_answered,
            }
        )
    return generation_results


def retrival_evaluation():
    """Evaluate retrieval results loaded from the retrieval JSONL file."""
    title_lookup = {}

    for record in vector_store.store.values():
        title = record["metadata"]["title"]
        source = record["metadata"]["source"]
        title_lookup[source] = title

    with open("eval/retrieval.jsonl", "r") as json_file:
        json_list = list(json_file)
    retrieval_results = []
    for json_str in json_list:
        highest_score = 0.0
        resource_id = 0.0
        record = json.loads(json_str)

        for title in title_lookup.items():
            match_value = fuzz.partial_ratio(title[1], record["question"])
            if match_value > highest_score:
                highest_score = match_value
                resource_id = title[0]

        result = vector_store.max_marginal_relevance_search(
            record["question"], 4, 20, 0.7
        )

        retrived_source_ids = []
        for doc in result:
            value = doc.metadata["source"].split(".")[0]
            retrived_source_ids.append(value)

        hit = record["dok_ordning_id"] in retrived_source_ids
        title_hit = record["dok_ordning_id"] == resource_id.split(".")[0]
        retrieval_results.append(
            {
                "id": record["id"],
                "hit": hit,
                "expected_source": record["dok_ordning_id"],
                "retrieved_sources": retrived_source_ids,
                "resource_id": resource_id.split(".")[0],
                "highest_score": highest_score,
                "title_hit": title_hit,
            }
        )
    return retrieval_results


response_format = {
    "reasoning": str,
    "asked_for_confirmation": bool,
    "contains_conflicting_statements": bool,
    "factually_accurate": bool,
}


def grade_printer(query_string: str, last_line: str):
    """Print the grading query and the model's final response line."""
    print("-------------------------------")
    print(query_string)
    print(last_line)
    print("-------------------------------")


def grade_answer(question: str, gold_answer: str, agent_answer: str) -> bool:
    """Generate a grading answer and return a bool if answer is living up to criteria"""
    query_string = create_question(question, gold_answer, agent_answer)
    response = init_chat_model(
        "ollama:qwen2.5:7b", temperature=0, response_format=response_format
    ).invoke(query_string)
    last_line = str(response.content).strip().splitlines()[-1]
    grade_printer(query_string, last_line)
    return last_line.startswith("GRADE: True")


def create_question(question: str, gold_answer: str, agent_answer: str):
    """Concrates the string with instruction, question, gold answer and agent answer for grade_answer"""
    # print(
    #    "Question:",
    #    question,
    #    "Gold_answer: ",
    #    gold_answer,
    #    "agent_answer:",
    #    agent_answer,
    # )
    return f"""
    ---------------------------------------
    Instructions: {CORRECTION_INSTRUCTIONS}, 
    Question: {question},
    Gold answer: {gold_answer}, 
    Agent answer: {agent_answer}
    ---------------------------------------
    """


results_retrival_eval = retrival_evaluation()
# print(results_retrival_eval)
for result in results_retrival_eval:
    print(result)

# Rapidfuzz for comparison - this is next on to do
# Optimize the dict with dict comprehension
