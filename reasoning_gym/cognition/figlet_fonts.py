from dataclasses import dataclass
from random import Random
from typing import Dict, Optional

import pyfiglet

from ..data.wordle_words import wordle_words
from ..factory import ProceduralDataset, register_dataset


@dataclass
class FigletFontConfig:
    """Configuration for FigletFont task generation"""

    static_word: Optional[str] = None
    static_font: Optional[str] = None
    space_letters: bool = True
    seed: Optional[int] = None
    size: int = 500


class FigletFontDataset(ProceduralDataset):
    """Generates FigletFont tasks"""

    def __init__(self, config: FigletFontConfig):
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

        word = self.config.static_word if self.config.static_word is not None else rng.choice(wordle_words).upper()
        if self.config.space_letters:
            render_word = " ".join(word)
        else:
            render_word = word

        # These ones are funky and probably aren't good for train/testing
        bad_fonts = [
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
        ]
        all_fonts = pyfiglet.FigletFont.getFonts()
        ok_fonts = list(filter(lambda x: x not in bad_fonts, all_fonts))
        chosen_font = self.config.static_font if self.config.static_font is not None else rng.choice(ok_fonts)
        figlet_render = pyfiglet.figlet_format(render_word, font=chosen_font)

        return {
            "question": rng.choice(self._prompt_templates).format(figlet_render=figlet_render),
            "answer": word,
            "metadata": {"font": chosen_font, "space_letters": self.config.space_letters},
        }

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
        """Determine if the solution provided solves the figlet task.

        The function awards 1.0 for a correct answer and 0.1 points for each correct letter in the correct position,
        with a maximum possible score of 1.0.

        Args:
            answer (Optional[str]): The user's answer.
            entry (Dict[str, any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        correct_word = entry["answer"]
        if not answer:
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


# Register the dataset
register_dataset("figlet_font", FigletFontDataset, FigletFontConfig)
