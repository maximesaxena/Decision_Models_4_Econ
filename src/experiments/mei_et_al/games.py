"""
A game: one decision from Mei, Xie, Yuan & Jackson (2024, PNAS). Decision models answer with one of its 10
bins (as a choice of amounts, or a score of levels); LLMs get Mei et al.'s original prompt.
"""

import json
import re
from dataclasses import dataclass

import numpy as np

# Jev's limits, in characters
MAX_STATE_CHARACTERS = 8000
MAX_INSTRUCTIONS_CHARACTERS = 1800
MAX_CRITERIA_CHARACTERS = 2000
MAX_OPTIONS = 10
MAX_LEVELS = 10


@dataclass(frozen=True)
class Game:
    name: str
    state: str  # the rules of the game, given to decision models
    question: str  # what the model decides, given to decision models
    original: str  # Mei et al.'s full prompt to ChatGPT, given to LLMs
    top: float  # largest amount
    unit: str = "${x:g}"  # how an amount is written

    def __post_init__(self):  # check every part fits Jev's limits
        assert len(self.state) <= MAX_STATE_CHARACTERS, f"{self.name}: state is {len(self.state)} characters"
        assert len(self.original) <= MAX_STATE_CHARACTERS, f"{self.name}: original prompt is {len(self.original)} characters"
        assert len(self.question) <= MAX_INSTRUCTIONS_CHARACTERS, f"{self.name}: question is {len(self.question)} characters"
        assert 2 <= len(self.options()) <= MAX_OPTIONS, f"{self.name}: {len(self.options())} options"
        assert len(self.levels()) <= MAX_LEVELS, f"{self.name}: {len(self.levels())} levels"
        for criteria in [self.options(), list(self.levels().values())]:
            characters = len(json.dumps(criteria))
            assert characters <= MAX_CRITERIA_CHARACTERS, f"{self.name}: criteria are {characters} characters"

    def options(self):
        """
        The answers a decision model chooses between: answer -> description.
        """
        return {f"{middle:g}": self.unit.format(x=middle) for middle in self.bin_middles()}

    def levels(self):
        """
        The score levels, one per bin: the bin's middle amount -> description.
        """
        edges = self.bin_edges()
        ends = [*(edges[1:-1] - 1), edges[-1]]  # the last bin includes the top amount
        return {f"{middle:g}": f"{self.unit.format(x=start)}–{self.unit.format(x=end)}"
                for middle, start, end in zip(self.bin_middles(), edges[:-1], ends)}

    def bin_edges(self):
        return np.linspace(0, self.top, 11)  # 10 bins

    def bin_middles(self):
        edges = self.bin_edges()
        return (edges[:-1] + edges[1:]) / 2

    def parse(self, response):
        """
        The answer in an LLM's response, read from [] as Mei et al. did; None if there is none.
        """
        matches = set(re.findall(r"\[(.*?)\]", response))
        if len(matches) != 1:
            return None
        answer = matches.pop().strip()
        try:
            amount = float(answer.removeprefix("$").replace(",", ""))
        except ValueError:
            return None
        return f"{amount:g}" if 0 <= amount <= self.top else None
