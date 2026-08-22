import json

from deepagents.backends import StateBackend
from langchain.chat_models import init_chat_model

from prompts import CORRECTION_INSTRUCTIONS

backend = StateBackend()

test_query = "Hvor mange ECTS er kandidattilvalget i latin normeret til, og hvornår trådte studieordningen i kraft?"
test_response = """

### Summary

**Key Rules and Requirements:**

- **ECTS Credits:** The Latin Candidature program offers a total of 45 ECTS credits.
- **Admission:** Admission to the Latin Candidature does not provide access to other degree programs.
- **Transitional Regulations:** As of September 1, 2026, the current study regulations from 2019 will be phased out. The following fagelements are transferred fromthe 2019 program to the 2026 program:
  - **Tekstanalyse og litteraturteori (10 ECTS):** This element is carried over without changes.
  - **Introduktion til romersk arkæologi (10 ECTS):** This fagelement cannot be equated with any course in the 2026 program, and there will be no transfer.
  - **Latinsk poesi (10 ECTS):** Similarly, this element cannot be equated with a course in the 2026 program, and there is no transfer.
  - **Introduktion til romersk arkæologi (10 ECTS) + Latinsk poesi (10 ECTS):** These two fagelements combine to form "Latinsk poesi i kontekst" (20 ECTS), with a pass mark.

### Study Structure:

- The program is structured into three semesters.
  - **Semester 1:** 
    - **Mandatory Courses:**
      - *Latinsk poesi i kontekst* (20 ECTS)
      - *Tekstanalyse og litteraturteori* (10 ECTS)
  - **Semester 2:**
    - **Mandatory Course:**
      - *Antikkens filosofi og dens efterliv* (15 ECTS)

### Important Notes:
- The study program is designed to be completed over two years, with each semester focusing on specific areas of Latin studies.
- No elective or supplementary courses are mentioned in the provided information.

### Conclusion:

This summary outlines the main points regarding ECTS credits, admission requirements, and the transitional changes from the 2019 regulations to the 2026 study program. The structure of the Latin Candidature is clearly defined with mandatory course work for each semester.

If you need more detailed information or specific courses, further details can be obtained by referring directly to the full document.
Based on the retrieved studieordning chunks:

- **ECTS Credits:** The Latin Candidature program offers a total of 45 ECTS credits. This includes mandatory courses such as "Latinsk poesi i kontekst" (20 ECTS) and "Tekstanalyse og litteraturteori" (10 ECTS), among others.

- **Admission to the Program:** Admission to the Latin Candidature does not grant access to other degree programs. The program is structured over three semesters, with each semester focusing on specific areas of Latin studies.

- **Transition from 2019 Regulations to 2026:** As of September 1, 2026, the current study regulations from 2019 will be phased out. Some fagelements are being transferred:
  - "Tekstanalyse og litteraturteori" (10 ECTS) remains unchanged.
  - "Introduktion til romersk arkæologi" and "Latinsk poesi" each carry 10 ECTS but cannot be equated with any course in the new program. They combine to form "Latinsk poesi i kontekst" (20 ECTS).
  - "Antikkens filosofi og dens efterliv" is a mandatory 15 ECTS course in the second semester.

The study program is designed to be completed over two years, with each semester focusing on specific areas of Latin studies. No elective or supplementary courses are mentioned in these chunks.

For more detailed information, you can refer directly to the full document referenced above.
"""

def generation_eval(user_query: str, llm_output: str):
    """Evaluate generation results loaded from the generation JSONL file."""
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
    response = init_chat_model("ollama:qwen2.5:7b").invoke(queryString)
    print(response)
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
