# -*- coding: utf-8 -*-


import json
from typing import Any, Dict, List

from reasoning_gym.data import get_data_file_path

from .const import (
    ALL_CONFIG_SIZE,
    ONOFFOFF_MAX_NON,
    ONOFFOFF_MAX_POTENTIAL,
    ONOFFOFF_MIN_NON,
    ONOFFOFF_MIN_POTENTIAL,
    ONONOFF_MAX_NON,
    ONONOFF_MAX_POTENTIAL,
    ONONOFF_MIN_NON,
    ONONOFF_MIN_POTENTIAL,
)


class BlicketView(object):
    def __init__(self, light_state="no"):
        # light state: "no" for no light, "off" for light off, "on" for light on
        self.objects = []
        self.light_state = light_state

    def add_objects(self, objects):
        for obj in objects:
            self.objects.append(obj)

    def remove_objects(self, objects):
        for obj in objects:
            self.objects.remove(obj)

    def __repr__(self):
        return "BlicketView(objects={}, light_state={})".format(self.objects, self.light_state)


class BlicketQuestion(object):
    def __init__(
        self,
        min_potential_blickets,
        max_potential_blickets,
        min_non_blickets,
        max_non_blickets,
        config_size,
        shuffle,
        rng,
    ):
        self.min_potential_blickets = min_potential_blickets
        self.max_potential_blickets = max_potential_blickets
        self.min_non_blickets = min_non_blickets
        self.max_non_blickets = max_non_blickets
        self.config_size = config_size
        self.shuffle = shuffle
        self.rng = rng

        potential_blicket_num = self.rng.randint(self.min_potential_blickets, self.max_potential_blickets)
        non_blicket_num = self.rng.randint(self.min_non_blickets, self.max_non_blickets)

        samples = self.rng.sample(list(range(self.config_size)), k=potential_blicket_num + non_blicket_num)

        self.blickets = []
        self.set_blickets = []
        self.potential_blickets = samples[:potential_blicket_num]
        self.non_blickets = samples[potential_blicket_num:]

        self.direct = []
        self.indirect = []
        self.screen_off = []

    def get_habituation_views(self):
        blicket_obj = self.potential_blickets[0]
        self.add_blicket(blicket_obj)

        non_blicket_obj = self.non_blickets[0]

        view_with_blicket = BlicketView("on")
        view_with_blicket.add_objects([blicket_obj])

        view_with_non_blicket = BlicketView("off")
        view_with_non_blicket.add_objects([non_blicket_obj])

        view_with_both = BlicketView("on")
        view_with_both.add_objects([blicket_obj, non_blicket_obj])

        habituation_views = [view_with_blicket, view_with_non_blicket, view_with_both]

        # bookkeeping for candidate choices
        self.direct.append(blicket_obj)
        self.screen_off.append(non_blicket_obj)

        return habituation_views

    def get_evidence_views(self):
        raise NotImplementedError("The parent class should not be used.")

    def get_views(self):
        habituation_views = self.get_habituation_views()
        evidence_views = self.get_evidence_views()
        if self.shuffle:
            self.rng.shuffle(habituation_views)
            self.rng.shuffle(evidence_views)

        return habituation_views + evidence_views

    def sanity_check(self):
        # no duplicates
        assert len(self.direct) == len(set(self.direct))
        assert len(self.indirect) == len(set(self.indirect))
        assert len(self.screen_off) == len(set(self.screen_off))
        # no intersection
        assert len(set(self.direct).intersection(self.indirect)) == 0
        assert len(set(self.direct).intersection(self.screen_off)) == 0
        assert len(set(self.indirect).intersection(self.screen_off)) == 0
        # completeness
        assert set(self.direct + self.indirect + self.screen_off) == set(self.blickets + self.non_blickets)

    def check_blickets(self, views):
        blickets = set()
        non_blickets = set()
        potential_blickets = set()

        direct = set()
        indirect = set()
        screen_off = set()

        on_views = [view for view in views if view.light_state == "on"]
        off_views = [view for view in views if view.light_state == "off"]

        assert len(on_views) + len(off_views) == len(views)

        for off_view in off_views:
            non_blickets.update(off_view.objects)
        for on_view in on_views:
            if len(on_view.objects) == 1:
                blickets.update(on_view.objects)
            else:
                diff_set = set(on_view.objects).difference(non_blickets)
                if len(diff_set) == 1:
                    blickets.update(diff_set)
        all_objects = set()
        for view in views:
            all_objects.update(view.objects)
        potential_blickets.update(all_objects.difference(non_blickets).difference(blickets))

        assert blickets == set(self.blickets)
        assert non_blickets == set(self.non_blickets)
        assert potential_blickets == set(self.potential_blickets)

        for on_view in on_views:
            if len(on_view.objects) == 1:
                direct.update(on_view.objects)
        on_view_objects = set()
        for on_view in on_views:
            on_view_objects.update(on_view.objects)
        off_view_objects = set()
        for off_view in off_views:
            off_view_objects.update(off_view.objects)

        direct.update(off_view_objects.difference(on_view_objects))
        screen_off.update(off_view_objects.intersection(on_view_objects))

        for on_view in on_views:
            if len(on_view.objects) > 1:
                diff_set = set(on_view.objects).difference(non_blickets)
                if len(diff_set) == 1 and not diff_set.issubset(direct):
                    indirect.update(diff_set)

        assert direct == set(self.direct)
        assert indirect == set(self.indirect)
        assert screen_off == set(self.screen_off)

        set_blickets = set()
        for on_view in on_views:
            on_diff_set = set(on_view.objects).difference(non_blickets)
            if len(on_diff_set.intersection(blickets)) == 0:
                potential_set_blicket = list(on_diff_set)
                potential_set_blicket.sort()
                set_blickets.add(tuple(potential_set_blicket))
        remove_set = set()
        for elem in set_blickets:
            for another_elem in set_blickets:
                elem_set = set(elem)
                another_elem_set = set(another_elem)
                if elem_set < another_elem_set:
                    remove_set.add(another_elem)
        set_blickets.difference_update(remove_set)

        assert set_blickets == set(self.set_blickets), "set_blickets:{}, self.set_blickets:{}".format(
            set_blickets, self.set_blickets
        )

    def check_labels(self, view, label):
        diff_set = set(view.objects).difference(self.non_blickets)
        if len(diff_set) == 0:
            assert label == 0
        else:
            blicket_inter = diff_set.intersection(self.blickets)
            if len(blicket_inter) > 0:
                assert label == 2
            else:
                if self.has_set_blicket(view):
                    assert label == 2
                else:
                    assert label == 1

    def has_set_blicket(self, view):
        for set_blicket in self.set_blickets:
            if set(set_blicket).issubset(view.objects):
                return True
        return False

    def union_sample(self, union, must_have_one=False):
        if must_have_one:
            first_set = self.rng.sample(union, k=1)
        else:
            first_num = self.rng.randint(1, len(union))
            first_set = self.rng.sample(union, k=first_num)
        second_set = list(set(union).difference(first_set))
        if len(second_set) > 0:
            additional_num_min = 0
        else:
            additional_num_min = 1
        additional_num = self.rng.randint(additional_num_min, len(first_set))
        additional_samples = self.rng.sample(first_set, k=additional_num)
        second_set += additional_samples

        return first_set, second_set

    def add_noise(self, view):
        # the first non blicket used in habituation and hence the skip
        noise_num = self.rng.randint(0, len(self.non_blickets) - 1)
        noise = self.rng.sample(self.non_blickets[1:], k=noise_num)
        view.add_objects(noise)

    def add_blicket(self, obj):
        self.potential_blickets.remove(obj)
        self.blickets.append(obj)

    def add_set_blicket(self, set_blicket):
        to_remove_set = []
        for already_set_blicket in self.set_blickets:
            if set(set_blicket) < set(already_set_blicket):
                to_remove_set.append(already_set_blicket)
            if set(already_set_blicket) < set(set_blicket):
                to_remove_set.append(set_blicket)
        self.set_blickets.append(set_blicket)
        for elem in to_remove_set:
            self.set_blickets.remove(elem)

    def generate_cause_questions(self, train, regime="IID"):

        def fixed_sum_sample(lower, upper, fixed_sum):
            values = [0] * len(lower)
            assert sum(upper) >= fixed_sum
            while True:
                residual = fixed_sum
                for i in range(len(values) - 1):
                    values[i] = self.rng.randint(lower[i], min(upper[i], residual))
                    residual = residual - values[i]
                if residual >= lower[-1] and residual <= upper[-1]:
                    values[-1] = residual
                    break
            assert sum(values) == fixed_sum
            return values

        # direct_sample, indirect_sample, screen_off_sample, potential_sample
        # these magic numbers and changes are to adjust dataset statistics
        constraint_lower = [0] * 3
        if regime == "Comp":
            # for Comp split
            constraint_upper = [len(self.direct), len(self.indirect), max(len(self.screen_off) - 2, 0)]
        if regime == "Sys":
            if train:
                # for Sys train split
                constraint_upper = [len(self.direct), len(self.indirect), max(len(self.screen_off) - 3, 0)]
            else:
                # for Sys val / test split
                constraint_upper = [len(self.direct), len(self.indirect), max(len(self.screen_off) - 1, 0)]
        if regime == "IID":
            # for IID split
            constraint_upper = [len(self.direct), len(self.indirect), max(len(self.screen_off) - 1, 0)]
        potential_sample_num = min(len(self.potential_blickets), 1)
        direct_sample_num, indirect_sample_num, screen_off_sample_num = fixed_sum_sample(
            constraint_lower, constraint_upper, 2 - potential_sample_num
        )
        direct_samples = self.rng.sample(self.direct, k=direct_sample_num)
        indirect_samples = self.rng.sample(self.indirect, k=indirect_sample_num)
        screen_off_samples = self.rng.sample(self.screen_off, k=screen_off_sample_num)
        potential_samples = self.rng.sample(self.potential_blickets, k=potential_sample_num)

        questions = []
        # label: 0 for light off, 1 for unknown, 2 for light up
        for direct_sample in direct_samples:
            assert direct_sample in self.blickets or direct_sample in self.non_blickets
            if direct_sample in self.blickets:
                label = 2
            else:
                label = 0
            cause_view = BlicketView("no")
            cause_view.add_objects([direct_sample])
            questions.append((cause_view, label, "direct"))
        for indirect_sample in indirect_samples:
            assert indirect_sample in self.blickets
            label = 2
            cause_view = BlicketView("no")
            cause_view.add_objects([indirect_sample])
            questions.append((cause_view, label, "indirect"))
        for screen_off_sample in screen_off_samples:
            assert screen_off_sample in self.non_blickets
            label = 0
            cause_view = BlicketView("no")
            cause_view.add_objects([screen_off_sample])
            questions.append((cause_view, label, "screen_off"))
        for potential_sample in potential_samples:
            assert potential_sample in self.potential_blickets
            label = 1
            cause_view = BlicketView("no")
            cause_view.add_objects([potential_sample])
            questions.append((cause_view, label, "potential"))

        if self.shuffle:
            self.rng.shuffle(questions)

        return questions

    def generate_intervention_questions(self, views):

        # on_views = [view for view in views if view.light_state == "on" and len(view.objects) >= 2]
        off_views = [view for view in views if view.light_state == "off"]

        # on_view = rng.sample(on_views, k=1)[0]
        # on_view_ref = views.index(on_view)
        off_view = self.rng.sample(off_views, k=1)[0]
        off_view_ref = views.index(off_view)

        questions = []

        all_possibilities = (
            list(set(self.blickets + self.potential_blickets + self.non_blickets).difference(off_view.objects))
            + self.set_blickets
        )
        # adjust weight during sample for better statistics
        all_possibilities += list(set(self.potential_blickets).difference(off_view.objects))
        possibilities = self.rng.sample(all_possibilities, k=2)
        for possibility in possibilities:
            if possibility in self.direct:
                q_type = "direct"
            elif possibility in self.indirect:
                q_type = "indirect"
            elif possibility in self.screen_off:
                q_type = "screen_off"
            elif possibility in self.potential_blickets:
                q_type = "potential"
            else:
                assert possibility in self.set_blickets
                q_type = "indirect"
            if possibility in self.blickets:
                label = 2
            elif possibility in self.potential_blickets:
                label = 1
            elif possibility in self.non_blickets:
                label = 0
            else:
                assert possibility in self.set_blickets
                label = 2
            intervention_view = BlicketView("no")
            intervention_view.add_objects(off_view.objects)
            if type(possibility) == tuple:
                possibility = list(possibility)
            else:
                possibility = [possibility]
            intervention_view.add_objects(possibility)
            questions.append((intervention_view, label, q_type, off_view_ref))

        if self.shuffle:
            self.rng.shuffle(questions)

        return questions


class OnOffOff(BlicketQuestion):
    def __init__(
        self,
        min_potential_blickets,
        max_potential_blickets,
        min_non_blickets,
        max_non_blickets,
        config_size,
        shuffle,
        rng,
    ):
        super(OnOffOff, self).__init__(
            min_potential_blickets,
            max_potential_blickets,
            min_non_blickets,
            max_non_blickets,
            config_size,
            shuffle,
            rng,
        )

    def get_evidence_views(self):
        # the first non blicket used in habituation and hence the skip
        first_off, second_off = self.union_sample(self.non_blickets[1:])

        first_off_view = BlicketView("off")
        first_off_view.add_objects(first_off)

        second_off_view = BlicketView("off")
        second_off_view.add_objects(second_off)

        on_view = BlicketView("on")

        on_list = self.potential_blickets[:]

        if len(on_list) == 1:
            self.add_blicket(on_list[0])
        else:
            set_blicket = on_list[:]
            set_blicket.sort()
            set_blicket = tuple(set_blicket)
            self.add_set_blicket(set_blicket)

        on_view.add_objects(on_list)

        self.add_noise(on_view)

        evidence_views = [on_view, first_off_view, second_off_view]

        # bookkeeping for candidate choices
        off_union_set = set(self.non_blickets[1:])
        on_set = set(on_view.objects)
        on_diff_set = on_set.difference(off_union_set)
        direct_set = list(off_union_set.difference(on_set))
        screen_off_set = list(off_union_set.intersection(on_set))

        self.direct.extend(direct_set)
        self.screen_off.extend(screen_off_set)

        if len(on_set) == 1:
            self.direct.extend(list(on_set))
        else:
            if len(on_diff_set) == 1:
                self.indirect.extend(list(on_diff_set))
        return evidence_views


class OnOnOff(BlicketQuestion):
    def __init__(
        self,
        min_potential_blickets,
        max_potential_blickets,
        min_non_blickets,
        max_non_blickets,
        config_size,
        shuffle,
        rng,
    ):
        super(OnOnOff, self).__init__(
            min_potential_blickets,
            max_potential_blickets,
            min_non_blickets,
            max_non_blickets,
            config_size,
            shuffle,
            rng,
        )

    def get_evidence_views(self):
        # the first non blicket used in habituation and hence the skip
        off_list = self.non_blickets[1:]

        off_view = BlicketView("off")
        off_view.add_objects(off_list)

        first_on, second_on = self.union_sample(self.potential_blickets)

        first_on_view = BlicketView("on")
        first_on_view.add_objects(first_on)

        second_on_view = BlicketView("on")
        second_on_view.add_objects(second_on)

        for on in [first_on, second_on]:
            if len(on) == 1:
                if on[0] not in self.blickets:
                    self.add_blicket(on[0])
        for on in [first_on, second_on]:
            if len(set(on).intersection(self.blickets)) == 0:
                set_blicket = on[:]
                set_blicket.sort()
                set_blicket = tuple(set_blicket)
                if set_blicket not in self.set_blickets:
                    self.add_set_blicket(set_blicket)

        self.add_noise(first_on_view)
        self.add_noise(second_on_view)

        evidence_views = [first_on_view, second_on_view, off_view]

        # bookkeeping for candidate choices
        on_union_set = set(first_on_view.objects).union(second_on_view.objects)
        off_set = set(off_view.objects)

        direct_set = list(off_set.difference(on_union_set))
        screen_off_set = list(off_set.intersection(on_union_set))
        self.direct.extend(direct_set)
        self.screen_off.extend(screen_off_set)

        for on_view in [first_on_view, second_on_view]:
            if len(on_view.objects) == 1:
                obj = on_view.objects[0]
                if obj not in self.direct:
                    self.direct.append(obj)
        for on_view in [first_on_view, second_on_view]:
            diff_set = set(on_view.objects).difference(off_set)
            if len(on_view.objects) > 1 and len(diff_set) == 1:
                obj = diff_set.pop()
                if obj not in self.indirect and obj not in self.direct:
                    self.indirect.append(obj)

        return evidence_views


def serialize(questions):
    question_list = []
    for question in questions:
        view_list = []
        for i in range(6):
            json_dict = {}
            json_dict["light_state"] = question[i].light_state
            json_dict["objects"] = question[i].objects
            view_list.append(json_dict)
        for i in range(6, 8):
            json_dict = {}
            json_dict["light_state"] = question[i][0].light_state
            json_dict["objects"] = question[i][0].objects
            json_dict["label"] = question[i][1]
            json_dict["type"] = question[i][2]
            view_list.append(json_dict)
        for i in range(8, 10):
            json_dict = {}
            json_dict["light_state"] = question[i][0].light_state
            json_dict["objects"] = question[i][0].objects
            json_dict["label"] = question[i][1]
            json_dict["type"] = question[i][2]
            json_dict["ref"] = question[i][3]
            view_list.append(json_dict)
        question_list.append(view_list)
    return question_list


def config_control(size, train, config_size, regime, rng):
    questions = []
    for _ in range(size // 2):
        blicket_machine = OnOffOff(
            ONOFFOFF_MIN_POTENTIAL, ONOFFOFF_MAX_POTENTIAL, ONOFFOFF_MIN_NON, ONOFFOFF_MAX_NON, config_size, True, rng
        )
        context_views = blicket_machine.get_views()

        blicket_machine.sanity_check()
        blicket_machine.check_blickets(context_views)

        cause_questions = blicket_machine.generate_cause_questions(train=train, regime=regime)
        for cause_question in cause_questions:
            blicket_machine.check_labels(cause_question[0], cause_question[1])
        intervention_questions = blicket_machine.generate_intervention_questions(context_views)
        for intervention_question in intervention_questions:
            blicket_machine.check_labels(intervention_question[0], intervention_question[1])
        questions.append(context_views + cause_questions + intervention_questions)
    for _ in range(size // 2):
        blicket_machine = OnOnOff(
            ONONOFF_MIN_POTENTIAL, ONONOFF_MAX_POTENTIAL, ONONOFF_MIN_NON, ONONOFF_MAX_NON, config_size, True, rng
        )
        context_views = blicket_machine.get_views()

        blicket_machine.sanity_check()
        blicket_machine.check_blickets(context_views)

        cause_questions = blicket_machine.generate_cause_questions(train=train, regime=regime)
        for cause_question in cause_questions:
            blicket_machine.check_labels(cause_question[0], cause_question[1])
        intervention_questions = blicket_machine.generate_intervention_questions(context_views)
        for intervention_question in intervention_questions:
            blicket_machine.check_labels(intervention_question[0], intervention_question[1])
        questions.append(context_views + cause_questions + intervention_questions)
    rng.shuffle(questions)
    return questions


def dist_control(size, train, regime, rng):
    questions = []
    for _ in range(size):
        if train:
            blicket_machine = OnOffOff(
                ONOFFOFF_MIN_POTENTIAL,
                ONOFFOFF_MAX_POTENTIAL,
                ONOFFOFF_MIN_NON,
                ONOFFOFF_MAX_NON,
                ALL_CONFIG_SIZE,
                True,
                rng,
            )
        else:
            blicket_machine = OnOnOff(
                ONONOFF_MIN_POTENTIAL,
                ONONOFF_MAX_POTENTIAL,
                ONONOFF_MIN_NON,
                ONONOFF_MAX_NON,
                ALL_CONFIG_SIZE,
                True,
                rng,
            )
        context_views = blicket_machine.get_views()

        blicket_machine.sanity_check()
        blicket_machine.check_blickets(context_views)

        cause_questions = blicket_machine.generate_cause_questions(train=train, regime=regime)
        for cause_question in cause_questions:
            blicket_machine.check_labels(cause_question[0], cause_question[1])
        intervention_questions = blicket_machine.generate_intervention_questions(context_views)
        for intervention_question in intervention_questions:
            blicket_machine.check_labels(intervention_question[0], intervention_question[1])
        questions.append(context_views + cause_questions + intervention_questions)
    rng.shuffle(questions)
    return questions


# Text translation functions
LIGHT_DICT_TEXT = ["off", "undetermined", "on"]
OBJECT_DICT_FILE_PATH = get_data_file_path("acre_objects.json")
with open(OBJECT_DICT_FILE_PATH, "r") as object_dict_file:
    OBJECT_DICT_TEXT = json.load(object_dict_file)


def get_example_text(objects, light_state):
    input_examples = []
    for object in objects:
        obj_desc = OBJECT_DICT_TEXT[str(object)]
        color = obj_desc["color"]
        shape = obj_desc["shape"]
        material = obj_desc["material"]
        # [color, material, shape]
        input_examples.append([color, material, shape])
    return {"input": input_examples, "output": light_state}


# Label has 0, 1, 2 integer which are indexed into the array ["off", "undetermined", "on"]
def get_trial_text(objects, label):
    # object_text = ""
    input_examples = []
    for object in objects:
        obj_desc = OBJECT_DICT_TEXT[str(object)]
        color = obj_desc["color"]
        shape = obj_desc["shape"]
        material = obj_desc["material"]
        # [color, material, shape]
        input_examples.append([color, material, shape])
    return {"input": input_examples, "output": LIGHT_DICT_TEXT[label]}


# Select translation functions
get_example = get_example_text
get_trial = get_trial_text
get_answer_list = lambda: LIGHT_DICT_TEXT


def final_parse(serialized_questions) -> List[Dict[str, Any]]:
    output_data = []
    for sample in serialized_questions:  # For each data in input (6 examples, 4 tests)
        # {"examples": [{"input": [[color, material, shape]], "output": "on"}], "question": {"input": [color, material, shape], "output", "on", "type": "direct"}}
        # examples, ideally is an array of dict(input, output)
        examples = []
        for entry in sample:
            if entry["light_state"] != "no":  # example
                example = get_example(entry["objects"], entry["light_state"])
                examples.append(example)
            else:  # test case
                # question is a dict of input and output
                question = get_trial(entry["objects"], entry["label"])
                output_data.append({"examples": examples, "question": question})

    return output_data
