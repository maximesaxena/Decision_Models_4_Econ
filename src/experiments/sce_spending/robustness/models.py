"""
The models of common.models with a variant's system prompt: the usual one, or none.
"""

from tqdm import tqdm

from common import models

SHOWN_CALLS = 2  # the warm-up and the first saved call


class GPTNoSystem(models.GPT):
    """
    common.models.GPT without the system prompt.
    """

    def respond(self, prompt):
        response = self.client.responses.create(model=self.model, input=prompt)
        usage = response.usage
        details = getattr(usage, "output_tokens_details", None)
        return {
            "response": response.output_text or "",
            "thinking": "",  # OpenAI does not return its reasoning
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,  # includes reasoning tokens
            "thinking_tokens": getattr(details, "reasoning_tokens", None) or 0,
        }


def load(name, system):
    """
    The model, loaded as usual; with system None, an LLM's system prompt is dropped. Decision models have none.
    """
    model = models.load(name)
    if system == models.SYSTEM_PROMPT or name in models.DECISION_MODELS:
        return model
    assert system is None, "a variant's system prompt is the usual one or none"
    if isinstance(model, models.Gemini):
        model.config = model.config.model_copy(update={"system_instruction": None})
    elif isinstance(model, models.GPT):
        model.__class__ = GPTNoSystem  # keeps the model's own client and settings
    else:
        raise ValueError(f"{name} has no system prompt to drop")
    return model


def show_calls(model, calls=SHOWN_CALLS):
    """
    The model, printing what an LLM hands its SDK (system prompt and prompt) on its first calls, as they are made.
    """
    if isinstance(model, models.Gemini):
        sdk, method = model.client.models, "generate_content"
    elif isinstance(model, models.GPT):
        sdk, method = model.client.responses, "create"
    else:
        return model
    call = getattr(sdk, method)
    shown = 0

    def showing(**kwargs):
        nonlocal shown
        if shown < calls:
            shown += 1
            if "config" in kwargs:  # Gemini
                system, prompt = kwargs["config"].system_instruction, kwargs["contents"]
            else:
                system, prompt = kwargs.get("instructions"), kwargs["input"]
            tqdm.write(f"--- {kwargs['model']}, call {shown}: system prompt {'NONE' if system is None else repr(system)}\n"
                       f"{prompt}\n---")
        return call(**kwargs)

    setattr(sdk, method, showing)
    return model
