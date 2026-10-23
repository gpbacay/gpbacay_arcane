"""Tests for Occipital: letter-level recall for Hippocampus (no TensorFlow needed)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpbacay_arcane.got import DocumentGraph
from gpbacay_arcane.hippocampus import Hippocampus
from gpbacay_arcane.occipital import Occipital, grams

TICKETS = [("I was charged twice", "billing"), ("refund my last invoice", "billing"),
           ("my parcel never arrived", "shipping"), ("track my delivery", "shipping")]
LABELS = ["billing", "shipping"]


def memory(occipital=None, **kwargs):
    h = Hippocampus(occipital=occipital, **kwargs)
    for text, label in TICKETS:
        h.remember(text, label)
    return h


def test_grams_are_padded_letter_trigrams_and_fourgrams():
    assert grams("Hi") == {" hi": 1, "hi ": 1, " hi ": 1}
    assert " par" in grams("parcel") and "cel " in grams("parcel")


def test_typos_find_neighbors_that_words_miss():
    typo = "my parcle nevr arived"
    assert memory().classify(typo, LABELS)["label"] is None  # no whole word in common but "my", a stopword
    out = memory(Occipital()).classify(typo, LABELS)
    assert out["label"] == "shipping"
    assert out["neighbors"][0]["text"] == "my parcel never arrived"


def test_it_still_abstains_on_requests_unlike_anything_stored():
    assert memory(Occipital()).classify("hello partner", LABELS)["label"] is None  # only " par" in common
    assert memory(Occipital(min_coverage=0.0)).react("hello partner")["neighbors"]  # the knob trades abstention away


def test_word_matches_still_win():
    out = memory(Occipital()).classify("I was charged twice for the invoice", LABELS)
    assert out["label"] == "billing"
    assert out["neighbors"][0]["text"] == "I was charged twice"


def test_the_letter_index_follows_forget_remember_and_eviction():
    h = memory(Occipital())
    h.forget("my parcel never arrived")
    assert all(n["text"] != "my parcel never arrived" for n in h.neighbors("my parcle nevr arived"))
    h.remember("my parcel never arrived", "billing")  # relabelled
    assert h.classify("my parcle nevr arived", LABELS)["label"] == "billing"

    capped = Hippocampus(occipital=Occipital(), max_size=1)
    capped.observe("my parcel never arrived", "shipping")
    capped.observe("refund my last invoice", "billing")  # evicts the parcel example
    assert all(n["label"] != "shipping" for n in capped.neighbors("my parcle nevr arived"))


def test_a_loaded_graph_brings_its_letters_back():
    graph = DocumentGraph.from_json(memory().graph.to_json())
    h = Hippocampus(graph=graph, occipital=Occipital())
    assert h.classify("my parcle nevr arived", LABELS)["label"] == "shipping"


def test_label_filter_applies_to_letter_matches():
    assert memory(Occipital()).neighbors("my parcle nevr arived", ["billing"]) == []  # too little in common
    hits = memory(Occipital(min_coverage=0.0, max_df=1.0)).neighbors("my parcle nevr arived", ["billing"])
    assert hits and all(h["label"] == "billing" for h in hits)
