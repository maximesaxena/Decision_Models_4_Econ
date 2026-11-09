"""
Every model, loaded by name with load(name).

Decision models have ask(state, question) -> (probability per answer, usage); the probabilities are empty on a
refusal. Questions use Jev's System One format:
- {"type": "choice", "instructions": ..., "criteria": {answer: description, ...}}
- {"type": "score", "instructions": ..., "criteria": [level 0, level 1, ...]}, answers keyed "0", "1", ...

LLMs have respond(prompt) -> one sampled response with its token counts.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Laya's limits on input length <- as I did this work this became increasingly obsolete but still kept out of precaution
PROMPT_TOKEN_LIMIT = 512
QUESTION_TOKEN_LIMIT = 192


QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3.5-4B")

# Qwen's recommended settings for each mode
QWEN_SAMPLING = {
    True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20},  # thinking
    False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20},  # not thinking
}
QWEN_MAX_NEW_TOKENS = 32768

GEMINI_MODEL = "gemini-3.8-flash"

GEMINI_ARGON_MODEL = "gemini-4-argon"

GEMMA_MODEL = "gemma-4-26b-a4b-it"  # through the Gemini API

LUNA_MODEL = "gpt-6-luna"

GPT_MODEL = "gpt-6-luna"

GPT_SOL_MODEL = "gpt-6-sol"

GLM_FLASH_MODEL = "perplexity/glm-5.3-flash"  

HAIKU_MODEL = "claude-haiku-5-5"
CLAUDE_MAX_TOKENS = 16000  # cap required for that API

PPLX_MODEL = "pplx-decider-v1.1-27b"

MAMMOTH_MODEL = "decision-models/baseten-mammoth-27b-latest" 

MERCURY_MODEL = "inception/mercury-decide"

MICROSOFT_MODEL = "microsoft/microsoft-decision-1"

CLEF_MODEL = "cloudflare/clef-omni"

LIQUID_MODEL = "liquid/d1-ultra"

SYSTEM_PROMPT = "You are a helpful assistant."  # Mei et al.'s system prompt


def prompt_tokens(tokenizer, state, question=None):
    """
    Tokens in the state plus, if given, the question's instructions and options.
    """
    texts = [state]
    if question:
        criteria = question["criteria"].values() if isinstance(question["criteria"], dict) else question["criteria"]
        texts += [question["instructions"], *criteria]
    return len(tokenizer("\n".join(texts))["input_ids"])


def device():
    """
    The first available of CUDA, Apple Silicon (MPS) and the CPU.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Laya:
    """
    Laya, run locally.
    """

    def __init__(self):
        import laya
        self.agent = laya.load("convaiinnovations/laya")

    @property
    def device(self):
        # read per call: a GPU memory fallback can leave Laya on the CPU
        return self.agent.device.type

    def ask(self, state, question):
        cpu_reruns, amp_failures = self.agent.cpu_fallback_count, self.agent._amp_failures
        result = self.agent.predict(state, {"answer": question})
        internal_retries = (self.agent.cpu_fallback_count - cpu_reruns) + int(self.agent._amp_failures == amp_failures + 1)
        usage = result["usage"]
        # Laya silently truncates inputs over its limits
        assert not usage["truncated"] and not usage["truncated_questions"], f"Laya truncated the input: {usage}"
        question_tokens = usage["input_tokens"] - usage["state_tokens"] - 1
        assert question_tokens <= QUESTION_TOKEN_LIMIT, f"Question and options are {question_tokens} tokens, over Laya's {QUESTION_TOKEN_LIMIT}"
        return result["answers"]["answer"]["probabilities"], {"input_tokens": usage["input_tokens"], "internal_retries": internal_retries}


class SystemOne:
    """
    A model behind TypeSafe's System One API: Jev (either provider) or a Kev server.
    """

    def __init__(self, model, base_url=None, api_key=None, device="api"):
        from typesafe_sdk import RetryPolicy, TypeSafeClient
        self.client = TypeSafeClient(model=model, base_url=base_url, api_key=api_key, retry=RetryPolicy(max_retries=0))
        self.device = device

    def ask(self, state, question):
        from typesafe_sdk import Choice, Score
        question_type = {"choice": Choice, "score": Score}[question["type"]]
        response = self.client.system_one(state=state, questions={"answer": question_type(**question)})
        # Jev keys score levels as strings
        probs = {str(answer): p for answer, p in response.answers["answer"].probabilities.items()}
        return probs, {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}


def jev():
    """
    Jev through TypeSafe's own API. The key is passed explicitly, as the SDK defaults to TYPESAFE_API_KEY
    (the third-party provider's).
    """
    api_key = os.getenv("JEV_API_KEY", "").strip()
    assert api_key, "jev needs JEV_API_KEY (a TypeSafe API key) in .env"
    return SystemOne("jev-latest", base_url=os.getenv("JEV_BASE_URL", "https://api.typesafe.ai"), api_key=api_key)


def jev_3rdparty():
    """
    Jev through the third-party provider jevmodel.org.
    """
    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    assert api_key, "jev-3rdparty needs TYPESAFE_API_KEY (a jevmodel.org key) in .env"
    return SystemOne("jev-latest", base_url=os.getenv("TYPESAFE_BASE_URL", "https://jevmodel.org"), api_key=api_key)


def mammoth():
    """
    Baseten's mammoth-1 through Baseten's System One API.
    """
    api_key = os.getenv("BASETEN_API_KEY", "").strip()
    assert api_key, "mammoth needs BASETEN_API_KEY in .env"
    return SystemOne(MAMMOTH_MODEL, base_url="https://inference.baseten.co", api_key=api_key)


class Luna:
    """
    GPT-6 Luna through OpenAI's Decisions API.
    """

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(max_retries=0)
        self.device = "api"

    def ask(self, state, question):
        if question["type"] == "choice":
            spec = {"choices": [{"value": answer, "description": description} for answer, description in question["criteria"].items()]}
        else:
            spec = {"levels": [{"label": str(level), "description": description} for level, description in enumerate(question["criteria"])]}
        decision = self.client.decisions.create(model=LUNA_MODEL, input=state, questions=[
            {"type": question["type"], "name": "answer", "instructions": question["instructions"], **spec}])
        answer = decision.answers[0]
        # a refusal has no probabilities; score levels come back as ints
        probs = {} if answer.type == "refusal" else {str(p.value): p.probability for p in answer.probabilities}
        # usage may be missing
        usage = getattr(decision, "usage", None)
        return probs, {"input_tokens": getattr(usage, "input_tokens", None), "output_tokens": getattr(usage, "output_tokens", None)}


class Pplx:
    """
    Perplexity's decision model through its Decisions API, which takes questions in System One format.
    """

    def __init__(self):
        import httpx
        api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
        assert api_key, "pplx needs PERPLEXITY_API_KEY in .env"
        self.client = httpx.Client(base_url="https://api.perplexity.ai", timeout=60, headers={"Authorization": f"Bearer {api_key}"})
        self.device = "api"

    def ask(self, state, question):
        response = self.client.post("/v1/decisions", json={"model": PPLX_MODEL, "state": state, "questions": {"answer": question}})
        response.raise_for_status()
        decision = response.json()
        probs = {str(answer): p for answer, p in decision["answers"]["answer"].get("probabilities", {}).items()}
        usage = decision.get("usage") or {}
        return probs, {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}


class OpenRouter:
    """
    A decision model through OpenRouter's (alpha) Decisions API, which takes questions in System One format.
    """

    def __init__(self, model):
        import httpx
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        assert api_key, f"{model} needs OPENROUTER_API_KEY in .env"
        self.client = httpx.Client(base_url="https://openrouter.ai", timeout=60, headers={"Authorization": f"Bearer {api_key}"})
        self.model = model
        self.device = "api"

    def ask(self, state, question):
        response = self.client.post("/api/alpha/decisions", json={"model": self.model, "state": state, "questions": {"answer": question}})
        response.raise_for_status()
        decision = response.json()
        probs = {str(answer): p for answer, p in decision["answers"]["answer"].get("probabilities", {}).items()}
        usage = decision.get("usage") or {}
        return probs, {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}


class AnyJev:
    """
    AnyJev on Qwen, run locally.
    """

    def __init__(self):
        from anyjev import Decider
        from anyjev.backends.hf import HFBackend
        self.device = device()
        self.decider = Decider(HFBackend(QWEN_MODEL, device=self.device), shared_prefix=False)

    def ask(self, state, question):
        from anyjev import Question
        if question["type"] == "choice":
            answers = list(question["criteria"])
            anyjev_question = Question.choice(question["instructions"], list(question["criteria"].values()))
        else:
            answers = [str(level) for level in range(len(question["criteria"]))]
            anyjev_question = Question.score(question["instructions"], levels=question["criteria"])
        decision = self.decider.decide(state, [anyjev_question])[0]
        return dict(zip(answers, decision.probs.tolist())), {"input_tokens": prompt_tokens(self.decider.backend.tokenizer, state, question)}


class Qwen:
    """
    Qwen as a chat model, with thinking on or off.
    """

    def __init__(self, thinking):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.thinking = thinking
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
        self.device = device()
        self.model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL, dtype=torch.bfloat16).to(self.device)
        self.end_think = self.tokenizer.convert_tokens_to_ids("</think>")

    def respond(self, prompt):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, do_sample=True, max_new_tokens=QWEN_MAX_NEW_TOKENS,
                                     **QWEN_SAMPLING[self.thinking])[0, inputs["input_ids"].shape[1]:].tolist()
        thinking_tokens = output.index(self.end_think) + 1 if self.end_think in output else (len(output) if self.thinking else 0)
        return {
            "response": self.tokenizer.decode(output[thinking_tokens:], skip_special_tokens=True).strip(),
            "thinking": self.tokenizer.decode(output[:thinking_tokens], skip_special_tokens=True).strip(),
            "input_tokens": prompt_tokens(self.tokenizer, prompt),
            "output_tokens": len(output),
            "thinking_tokens": thinking_tokens,
        }


class Gemini:
    """
    Gemini (or another model, such as Gemma) through Google's Gemini API. The key defaults to GEMINI_API_KEY.
    """

    def __init__(self, thinking_level=None, model=GEMINI_MODEL, api_key=None):
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.device = "api"
        thinking_config = types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None
        self.config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, thinking_config=thinking_config)

    def respond(self, prompt):
        response = self.client.models.generate_content(model=self.model, contents=prompt, config=self.config)
        usage = response.usage_metadata
        return {
            "response": response.text or "",
            "thinking": "",  
            "input_tokens": usage.prompt_token_count,
            "output_tokens": (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
            "thinking_tokens": usage.thoughts_token_count or 0,
        }


class GPT:
    """
    A GPT model through OpenAI's Responses API, or another model through an API that speaks it.
    """

    def __init__(self, model=GPT_MODEL, base_url=None, api_key=None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.model = model
        self.device = "api"

    def respond(self, prompt):
        response = self.client.responses.create(model=self.model, instructions=SYSTEM_PROMPT, input=prompt)
        usage = response.usage
        details = getattr(usage, "output_tokens_details", None)
        return {
            "response": response.output_text or "",
            "thinking": "",  
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,  # includes reasoning tokens
            "thinking_tokens": getattr(details, "reasoning_tokens", None) or 0,
        }


def gemma():
    """
    Gemma through the Gemini API, on its own key.
    """
    api_key = os.getenv("GEMMA_API_KEY", "").strip()
    assert api_key, "gemma needs GEMMA_API_KEY (a Gemini API key) in .env"
    return Gemini(model=GEMMA_MODEL, api_key=api_key)


def glm_flash():
    """
    GLM-5.3-Flash through Perplexity's Agent API, which speaks OpenAI's Responses API. 
    """
    api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    assert api_key, "glm-flash needs PERPLEXITY_API_KEY in .env"
    return GPT(GLM_FLASH_MODEL, base_url="https://api.perplexity.ai/v1", api_key=api_key)


class Claude:
    """
    A Claude model through Anthropic's Messages API, with its default settings (for Haiku 5.5, adaptive thinking at
    medium effort; sampling parameters are rejected). The key comes from ANTHROPIC_API_KEY.
    """

    def __init__(self, model=HAIKU_MODEL):
        from anthropic import Anthropic
        self.client = Anthropic(max_retries=0)
        self.model = model
        self.device = "api"

    def respond(self, prompt):
        response = self.client.messages.create(model=self.model, max_tokens=CLAUDE_MAX_TOKENS, system=SYSTEM_PROMPT,
                                               messages=[{"role": "user", "content": prompt}])
        usage = response.usage
        details = getattr(usage, "output_tokens_details", None)
        return {
            "response": "".join(block.text for block in response.content if block.type == "text").strip(),
            "thinking": "",  
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,  # includes thinking tokens
            "thinking_tokens": getattr(details, "thinking_tokens", None) or 0,
        }


DECISION_MODELS = {
    "laya": Laya,
    "jev": jev,
    "jev-3rdparty": jev_3rdparty,
    "kev": lambda: SystemOne("kev-latest", base_url=os.getenv("KEV_BASE_URL", "http://127.0.0.1:8009"), api_key="local", device="server"),
    "anyjev": AnyJev,
    "luna": Luna,
    "pplx": Pplx,
    "mammoth": mammoth,
    "mercury": lambda: OpenRouter(MERCURY_MODEL),
    "d1-ultra": lambda: OpenRouter(LIQUID_MODEL),
    "microsoft": lambda: OpenRouter(MICROSOFT_MODEL),
    "clef": lambda: OpenRouter(CLEF_MODEL),
}

LLMS = {
    "qwen": lambda: Qwen(thinking=True),
    "qwen-nothink": lambda: Qwen(thinking=False),
    "gemini": lambda: Gemini(),
    "gemini-lowthink": lambda: Gemini(thinking_level="low"),
    "gpt": GPT,
    "gpt-sol": lambda: GPT(GPT_SOL_MODEL),
    "gemma": gemma,
    "glm-flash": glm_flash,
    "gemini-argon": lambda: Gemini(model=GEMINI_ARGON_MODEL),
    "haiku": Claude,
}

MODELS = DECISION_MODELS | LLMS

# Dollars per million (input, output) tokens
PRICES = {"jev": (0.042, 0.0), "luna": (0.10, 0.0), "gemini": (0.75, 3.75), "gemini-lowthink": (0.75, 3.75), "jev-3rdparty": (0.66,0.0), "pplx": (0.02, 0.0), "gpt": (0.1, 0.5), "gpt-sol": (2.0, 10.0),
          "mercury": (0.02,0.0), "d1-ultra": (0.042,0.0), "microsoft": (0.042,0.0), "clef": (0.15,0), "glm-flash": (0.15, 0.50),
          "haiku": (0.10, 0.50), "gemini-argon": (2.0, 10.0), "mammoth":(0.1,0), "gemma": (0.0, 0.0)}  
LOCAL_DEVICES = {"cuda", "mps", "cpu", "server"}


def load(name):
    return MODELS[name]()
