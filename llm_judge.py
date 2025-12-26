
import html
import re
import json
import time
import pandas as pd
import requests
from typing import Optional, Dict, Any, List


# Prompt used as SYSTEM input for our LLM judge
JUDGE_PROMPT = """
You are an expert financial evaluator and LLM judge. Your role is to assess the quality of a system-generated answer to a user question based on four dimensions:

1. **Precision** – Is the answer clear, specific, and accurate?
2. **Faithfulness** – Does the answer remain factually correct and grounded in the source text?
3. **Comprehensiveness** – Does the answer fully capture the core information and context?
4. **Relevance** – Is the answer aligned with the main topic and intent of the user question?

### Instructions:
- Rate each dimension on a scale of **1 to 5** (1 = very poor, 5 = excellent).
- Provide a **final overall score** (average or weighted if needed).
- Explain your reasoning for each dimension clearly and concisely.
- Output your evaluation in **valid JSON format** as shown below.

### Output Format:
<ANSWER>
{
  "precision_score": <integer>,
  "faithfulness_score": <integer>,
  "comprehensiveness_score": <integer>,
  "relevance_score": <integer>,
  "overall_score": <integer>,
  "reasoning": {
    "precision": "<your explanation>",
    "faithfulness": "<your explanation>",
    "comprehensiveness": "<your explanation>",
    "relevance": "<your explanation>",
    "overall": "<summary of judgment>"
  }
}
</ANSWER>

### Additional Guidelines:
- Be **objective and consistent** in scoring.
- If information is missing or ambiguous, deduct points and explain why.
- Do not include any extra text outside the JSON block.
<ANSWER>{}</ANSWER>
""".strip()

TEST_ORIGIN_TEXT = """
Apple There's little question that Apple (NASDAQ: AAPL) has become one of the most successful companies in history. 
It was clear that CEO Jensen Huang had a knack for skating to where the puck was going -- recognizing technology trends on the fly and adapting Nvidia's processors and the accompanying software to meet that need. 
The company, which began as a local online auction site, has evolved into the largest e-commerce and payments ecosystem in Latin America, serving 18 countries in the region.
""".strip()

# Generated sample from Copilot based on the origin text for evaluating the output
TEST_TRIPLET = {
  "triples": [
    {
      "subject": "Apple",
      "predicate": "has become",
      "object": "one of the most successful companies in history",
      "evidence": "Apple There's little question that Apple (NASDAQ: AAPL) has become one of the most successful companies in history.",
      "subject_type": "Organization",
      "object_type": "Descriptor",
      "confidence": 0.95
    },
    {
      "subject": "Apple",
      "predicate": "ticker symbol",
      "object": "NASDAQ: AAPL",
      "evidence": "Apple (NASDAQ: AAPL)",
      "subject_type": "Organization",
      "object_type": "TickerSymbol",
      "confidence": 0.99
    },
    {
      "subject": "Jensen Huang",
      "predicate": "role",
      "object": "CEO",
      "evidence": "It was clear that CEO Jensen Huang had a knack...",
      "subject_type": "Person",
      "object_type": "Role",
      "confidence": 0.9
    },
    {
      "subject": "Jensen Huang",
      "predicate": "recognized",
      "object": "technology trends on the fly",
      "evidence": "recognizing technology trends on the fly",
      "subject_type": "Person",
      "object_type": "Concept",
      "confidence": 0.85
    },
    {
      "subject": "Jensen Huang",
      "predicate": "adapted",
      "object": "Nvidia's processors and software to meet technology needs",
      "evidence": "adapting Nvidia's processors and the accompanying software to meet that need.",
      "subject_type": "Person",
      "object_type": "Product/Software",
      "confidence": 0.88
    },
    {
      "subject": "Nvidia",
      "predicate": "has products",
      "object": "processors and accompanying software",
      "evidence": "adapting Nvidia's processors and the accompanying software",
      "subject_type": "Organization",
      "object_type": "Product/Software",
      "confidence": 0.8
    },
    {
      "subject": "Unnamed company",
      "predicate": "began as",
      "object": "local online auction site",
      "evidence": "The company, which began as a local online auction site,",
      "subject_type": "Organization",
      "object_type": "Activity",
      "confidence": 0.75
    },
    {
      "subject": "Unnamed company",
      "predicate": "evolved into",
      "object": "largest e-commerce and payments ecosystem in Latin America",
      "evidence": "has evolved into the largest e-commerce and payments ecosystem in Latin America,",
      "subject_type": "Organization",
      "object_type": "OrganizationDescriptor",
      "confidence": 0.82
    },
    {
      "subject": "Largest e-commerce and payments ecosystem in Latin America",
      "predicate": "serves",
      "object": "18 countries in the region",
      "evidence": "serving 18 countries in the region.",
      "subject_type": "OrganizationDescriptor",
      "object_type": "GeographicScope",
      "confidence": 0.8
    }
  ],
  "notes": [
    "The text refers to 'CEO Jensen Huang' in the context of Nvidia, implying the role at Nvidia.",
    "The final paragraph describes an unnamed company; if the source is MercadoLibre (commonly matching this description), its name is not explicitly stated in the provided text, so we retain 'Unnamed company' to avoid assumptions."
  ]
}


# -------------------------- LM Studio Client -------------------------- #
class LMStudioLLM:
    """
    Minimal, testable client for LM Studio's REST API with a callable interface.

    Usage:
        llm = LMStudioLLM(
            model="openai/gpt-oss-20b",
            base_url="http://localhost:1234",
            timeout=30,
            temperature=0.2,
            max_tokens=512,
        )
        output_text = llm(prompt_text)
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:1234",
        timeout: float = 30.0,
        temperature: float = 0.2,
        max_tokens: int = 512,
        session: Optional[requests.Session] = None,
        system_message: Optional[str] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session = session or requests.Session()
        self.system_message = system_message  # If None, you can pass system content per call

        self._chat_url = f"{self.base_url}/v1/chat/completions"

    def __call__(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call the LM Studio chat endpoint and return the assistant content as string.
        """
        headers = {"Content-Type": "application/json"}
        messages = []

        # Prefer explicitly provided system prompt; fall back to self.system_message
        sys_content = system_prompt if system_prompt is not None else self.system_message
        if sys_content:
            messages.append({"role": "system", "content": sys_content})

        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        resp = self.session.post(self._chat_url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"LM Studio HTTP {resp.status_code}: {resp.text}")

        raw = resp.json()
        try:
            return raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return ""


# -------------------------- Judge Module -------------------------- #
class LLM_Judge:
    def __init__(self, llm_callable: LMStudioLLM):
        """
        Args:
            llm_callable: A callable that accepts (prompt: str, system_prompt?: str) and returns a string.
        """
        self.llm = llm_callable
        self.prompt = JUDGE_PROMPT
        self.columns = [
            "precision_score",
            "faithfulness_score",
            "comprehensiveness_score",
            "relevance_score",
            "overall_score",
            "reasoning",
        ]
        self.result = pd.DataFrame(columns=self.columns)


    def _extract_raw_output(self, raw_output: str) -> Optional[str]:
        """
        Extract JSON block between <ANSWER> ... </ANSWER>.
        - Unescapes HTML entities first.
        - Falls back to extracting the first JSON object if tags are missing.
        """
        if not raw_output:
            return None

        text = html.unescape(raw_output)

        # Primary: look for tags
        start = text.find("<ANSWER>")
        end = text.find("</ANSWER>")
        if start != -1 and end != -1 and end > start:
            return text[start + len("<ANSWER>"): end].strip()

        # Secondary: try to extract first JSON object via brace matching
        # This handles cases where the model omits the tags but returns pure JSON.
        first_brace = text.find("{")
        if first_brace != -1:
            depth = 0
            for i in range(first_brace, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[first_brace:i+1]
                        candidate = candidate.strip()
                        # Sanity check for JSON-like content
                        if candidate.startswith("{") and candidate.endswith("}"):
                            return candidate

        # Tertiary: regex-based attempt (less reliable but sometimes useful)
        m = re.search(r"\{(?:[^{}]|(?R))*\}", text)  # nested braces not fully supported in Python regex
        if m:
            return m.group(0).strip()

        return None


    def _build_eval_prompt(self, origin_text: str, system_answer_json: Dict[str, Any]) -> str:
        """
        Build the user message that provides context for the judge.
        """
        return (
            "User Question: Extract subject–predicate–object triples from the source text.\n\n"
            "Source Text:\n"
            f"{origin_text}\n\n"
            "System-Generated Answer (JSON):\n"
            f"{json.dumps(system_answer_json, ensure_ascii=False, indent=2)}\n\n"
            "Please evaluate the System-Generated Answer strictly based on the Source Text.\n"
            "Output ONLY the JSON inside <ANSWER>...</ANSWER> tags, per the system instructions."
        )

    def evaluate(self, origin_text: str, relevant_triplet: Dict[str, Any]) -> pd.DataFrame:
        """
        Evaluate the entire generated triple set once and store the JSON result.
        Returns the current results DataFrame.
        """
        user_prompt = self._build_eval_prompt(origin_text, relevant_triplet)
        raw_output = self.llm(user_prompt, system_prompt=self.prompt)
        answer_block = self._extract_raw_output(raw_output)

        if answer_block is None:
            raise ValueError("Could not find <ANSWER>...</ANSWER> block in model output.")

        try:
            evaluation = json.loads(answer_block)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON from model output: {e}\nRaw: {answer_block}")

        # Normalize reasoning into a single string for DataFrame or keep dict
        if "reasoning" in evaluation and isinstance(evaluation["reasoning"], dict):
            # Keep as dict, store in the 'reasoning' column
            row = {
                "precision_score": evaluation.get("precision_score"),
                "faithfulness_score": evaluation.get("faithfulness_score"),
                "comprehensiveness_score": evaluation.get("comprehensiveness_score"),
                "relevance_score": evaluation.get("relevance_score"),
                "overall_score": evaluation.get("overall_score"),
                "reasoning": evaluation["reasoning"],
            }
        else:
            # Fallback: place the entire evaluation in reasoning
            row = {
                "precision_score": evaluation.get("precision_score"),
                "faithfulness_score": evaluation.get("faithfulness_score"),
                "comprehensiveness_score": evaluation.get("comprehensiveness_score"),
                "relevance_score": evaluation.get("relevance_score"),
                "overall_score": evaluation.get("overall_score"),
                "reasoning": evaluation.get("reasoning"),
            }

        self.result = pd.concat([self.result, pd.DataFrame([row])], ignore_index=True)
        return self.result

    def evaluate_per_triple(self, origin_text: str, relevant_triplet: Dict[str, Any]) -> pd.DataFrame:
        """
        Optional: Evaluate each triple individually and append all results.

        Returns:
            DataFrame with one row per triple evaluation.
        """
        triples: List[Dict[str, Any]] = relevant_triplet.get("triples", [])
        if not triples:
            raise ValueError("No 'triples' found in relevant_triplet.")

        rows = []
        for i, triple in enumerate(triples, start=1):
            per_triple_answer = {"triple": triple}
            user_prompt = (
                "User Question: Evaluate the correctness of a single extracted triple against the source text.\n\n"
                "Source Text:\n"
                f"{origin_text}\n\n"
                "Triple Under Review (JSON):\n"
                f"{json.dumps(per_triple_answer, ensure_ascii=False, indent=2)}\n\n"
                "Please evaluate strictly based on the Source Text.\n"
                "Output ONLY the JSON inside <ANSWER>...</ANSWER> tags, per the system instructions."
            )

            raw_output = self.llm(user_prompt, system_prompt=self.prompt)
            answer_block = self._extract_raw_output(raw_output)
            if answer_block is None:
                # Skip but record a placeholder row
                rows.append({
                    "precision_score": None,
                    "faithfulness_score": None,
                    "comprehensiveness_score": None,
                    "relevance_score": None,
                    "overall_score": None,
                    "reasoning": {"overall": f"Missing <ANSWER> tags for triple #{i}."}
                })
                continue

            try:
                evaluation = json.loads(answer_block)
                rows.append({
                    "precision_score": evaluation.get("precision_score"),
                    "faithfulness_score": evaluation.get("faithfulness_score"),
                    "comprehensiveness_score": evaluation.get("comprehensiveness_score"),
                    "relevance_score": evaluation.get("relevance_score"),
                    "overall_score": evaluation.get("overall_score"),
                    "reasoning": evaluation.get("reasoning"),
                })
            except json.JSONDecodeError:
                rows.append({
                    "precision_score": None,
                    "faithfulness_score": None,
                    "comprehensiveness_score": None,
                    "relevance_score": None,
                    "overall_score": None,
                    "reasoning": {"overall": f"JSON parse error for triple #{i}."}
                })

        df = pd.DataFrame(rows)
        self.result = pd.concat([self.result, df], ignore_index=True)
        return self.result

    def return_results(self) -> pd.DataFrame:
        return self.result


# -------------------------- Example Usage -------------------------- #
if __name__ == "__main__":
    # 1) Instantiate LM Studio client (adjust model name to what you've loaded in LM Studio)
    llm = LMStudioLLM(
        model="openai/gpt-oss-20b",
        base_url="http://localhost:1234", # Replace with the host instance of LMStudio, this is the default
        temperature=0.1,
        max_tokens=512,
    )

    # 2) Create Judge
    judge = LLM_Judge(llm)

    # 3a) Evaluate the whole answer (triples JSON)
    try:
        results_df = judge.evaluate(TEST_ORIGIN_TEXT, TEST_TRIPLET)
        print("Whole-answer evaluation:")
        print(results_df.tail(1))
    except Exception as e:
        print("Evaluation error:", e)

    # 3b) (Optional) Evaluate per triple
    try:
        per_triple_df = judge.evaluate_per_triple(TEST_ORIGIN_TEXT, TEST_TRIPLET)
        print("\nPer-triple evaluations (appended):")
        print(per_triple_df.tail(len(TEST_TRIPLET["triples"])))
    except Exception as e:
        print("Per-triple evaluation error:", e)

