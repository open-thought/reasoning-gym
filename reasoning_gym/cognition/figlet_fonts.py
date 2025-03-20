import json
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import pyfiglet

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..data import get_data_file_path
from ..factory import ProceduralDataset, register_dataset

# These ones are funky and probably aren't good for train/testing
BAD_FONTS = [
    "pyramid",
    "runyc",
    "assalt_m",
    "term",
    "tengwar",
    "heart_right",
    "faces_of",
    "heroboti",
    "hieroglyphs",
    "rainbow_",
    "notie_ca",
    "ghost",
    "rampage_",
    "atc_____",
    "pacos_pe",
    "mad_nurs",
    "icl-1900",
    "joust___",
    "dcs_bfmo",
    "letter_w",
    "flyn_sh",
    "fun_face",
    "morse2",
    "tecrvs__",
    "ntgreek",
    "tsalagi",
    "etcrvs__",
    "faces_of",
    "future_8",
    "efti_robot",
    "danc4",
    "p_s_h_m_",
    "smkeyboard",
    "konto",
    "odel_lak",
    "courb",
    "jerusalem",
    "nfi1____",
    "keyboard",
    "konto_slant" "rot13",
    "mirror",
    "katakana",
    "cards",
    "eftichess",
    "heart_left",
    "trashman",
    "morse",
    "eftipiti",
    "smtengwar",
    "e__fist_",
    "mike",
    "bear",
    "hills___",
    "rotated",
    "wow",
    "eftipiti",
    "relief2",
    "mshebrew210",
    "kik_star",
    "puzzle",
    "p_skateb",
    "hypa_bal",
    "tomahawk",
    "timesofl",
    "moscow",
    "cola",
    "baz__bil",
    "stencil1",
    "battlesh",
    "tsn_base",
    "kgames_i",
    "binary",
    "greek",
    "mnemonic",
    "panther_",
    "b1ff",
    "c_consen",
    "horizontal_right",
    "dwhistled",
    "hex",
    "flipped",
    "high_noo",
    "patorjk-hex",
    "amc_3_liv1",
    "gauntlet",
    "cybersmall",
    "octal",
    "js_cursive",
    "battle_s",
    "deep_str",
    "rally_s2",
    "convoy__",
    "atc_gran",
    "grand_pr",
    "ivrit",
    "rammstein",
    "horizontal_left",
    "eftiwall",
    "decimal",
    "goofy",
    "rot13",
    "konto_slant",
    "subteran",
    "rally_sp",
    "charset_",
]
ALL_FONTS = pyfiglet.FigletFont.getFonts()
OK_FONTS = list(filter(lambda x: x not in BAD_FONTS, ALL_FONTS))

DATASET_NAME = "figlet_font"


@dataclass
class FigletFontConfig:
    """Configuration for FigletFont task generation"""

    static_word: Optional[str] = None
    static_font: Optional[str] = None
    min_word_len: int = 3
    max_word_len: int = 7
    space_letters: bool = True
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        assert self.min_word_len > 0, "min_word_len must be greater than 0"
        assert self.min_word_len <= self.max_word_len, "min_word_len must be less than or equal to max_word_len"
        if self.static_word:
            assert len(self.static_word) > 0, "static_word must have at least one character"
        if self.static_font:
            assert len(self.static_font) > 0, "static_font must have at least one character"
            assert self.static_font in OK_FONTS, f"static_font must be one of {OK_FONTS}"


class FigletFontDataset(ProceduralDataset):
    """Generates FigletFont tasks"""

    def __init__(self, config: FigletFontConfig):
        with get_data_file_path("anagrams.jsonl").open() as f:
            self.words = [
                word
                for line in f
                for word in json.loads(line)["words"]
                if config.min_word_len <= len(word) <= config.max_word_len
            ]
            assert len(self.words) > 0, "No words found in the dataset with the specified length range"

        self._prompt_templates = [
            "What word does this say?\n\n{figlet_render}",
            "Please read the following figlet font:\n\n{figlet_render}",
        ]
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single FigletFont task

        Returns:
            dict with keys:
                - question: str, the task description with figlet string
                - answer: str, the figlet encoded word
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        word = self.config.static_word if self.config.static_word is not None else rng.choice(self.words).upper()
        if self.config.space_letters:
            render_word = " ".join(word)
        else:
            render_word = word

        chosen_font = self.config.static_font if self.config.static_font is not None else rng.choice(OK_FONTS)
        figlet_render = pyfiglet.figlet_format(render_word, font=chosen_font)

        return {
            "question": rng.choice(self._prompt_templates).format(figlet_render=figlet_render),
            "answer": word,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "font": chosen_font,
                "space_letters": self.config.space_letters,
                "difficulty": {
                    "word_len": (self.config.min_word_len, self.config.max_word_len),
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the figlet task.

        The function awards 1.0 for a correct answer and 0.1 points for each correct letter in the correct position,
        with a maximum possible score of 1.0.

        Args:
            answer (Optional[str]): The user's answer.
            entry (dict[str, Any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        correct_word = entry["answer"]
        if not isinstance(answer, str):
            return 0.0  # No answer given

        # Normalize case
        answer = answer.replace(" ", "").strip().lower()
        correct_word = correct_word.strip().lower()

        if answer == correct_word:
            return 1.0  # Correct!

        # Calculate similarity
        correct_count = sum(1 for a, b in zip(answer, correct_word) if a == b)
        max_length = max(len(correct_word), len(answer))

        # Compute a partial score
        score = min(correct_count * 0.1, 1.0)

        return score


class FigletFontCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(FigletFontCurriculum.__name__, FigletFontConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="word_len",
                levels=[3, 5, 10, 15, 20, 30],
                default_level=0,
                description="The length of the word to be displayed",
                lower_field_name="min_word_len",
                upper_field_name="max_word_len",
                ensure_interval=True,
            ),
        )


# Register the dataset
register_dataset(DATASET_NAME, FigletFontDataset, FigletFontConfig, FigletFontCurriculum)
