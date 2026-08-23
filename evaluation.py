import json

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

from index import agent
from prompts import CORRECTION_INSTRUCTIONS


def generation_eval():
    """Evaluate generation results loaded from the generation JSONL file to evaulate output quality."""

    with open("eval/generation.jsonl", "r") as json_file:
        json_list = list(json_file)
        results = []
    for json_str in json_list:
        record = json.loads(json_str)
        result = agent.invoke({"messages": [HumanMessage(content=record["question"])]})
        llm_output = [message.text for message in result["messages"] if message.text]
        # print("LLM_OUPUT: ", llm_output)
        grade_answered = grade_answer(
            record["question"], record["gold_answer"], llm_output
        )
        results.append(
            {
                "id": record["id"],
                "grade": grade_answered,
            }
        )
    return results


def retrival_evaluation():
    """Evaluate retrieval results loaded from the retrieval JSONL file."""
    with open("eval/retrieval.jsonl", "r") as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)


def grade_answer(question: str, gold_answer: str, agent_answer: str) -> bool:
    """Generate a grading answer and return a bool if answer is living up to criteria"""
    queryString = createQuestion(question, gold_answer, agent_answer)
    response = init_chat_model("ollama:qwen2.5:7b").invoke(queryString)
    last_line = str(response.content).strip().splitlines()[-1]
    return last_line.startswith("GRADE: True")


def createQuestion(question: str, gold_answer: str, agent_answer: str):
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


results = generation_eval()
print(results)

# Need to fix temperature
# Need to fix prompts to answers like "Kan du lade mig udføre dette - etc." will not happen. Turns should not be allowed
# I might soon be running into problems with model size, since qwen2.5 7b is not that big and does not have a large context window for these kinds of tasks
