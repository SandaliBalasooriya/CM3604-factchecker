# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        raise Exception("Not implemented")

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        # return something


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold: float = 0.5):
        # Threshold is kept as a field so we can tune it later from one place
        # rather than hunting through the predict() body. Default 0.5 is a
        # starting guess; we will sweep this once the baseline runs end-to-end.
        self.threshold = threshold
    def predict(self, fact: str, passages: List[dict]) -> str:
        # Step 1: break the fact into a set of words.
        # Lowercase so "American" and "american" count as the same token.
        # Using a set rather than a list means repeated words only count once,
        # which matches the standard recall definition on word types.
        fact_words = set(fact.lower().split())

        # Step 2: for each passage, compute the recall of the fact's words.
        # We take the max rather than the mean because a fact only needs ONE
        # supporting passage. Averaging would unfairly punish a fact that is
        # strongly supported by passage 1 but unrelated to passages 2 and 3.
        best_score = 0.0
        for passage in passages:
            # passage is a dict like {"title": "...", "text": "..."} - we only
            # need "text". The brief tells us the title is not useful here.
            passage_words = set(passage["text"].lower().split())

            # Recall = how many of the fact's words appear in this passage,
            # divided by the total number of fact words. Range: 0.0 to 1.0.
            # Guard against division by zero in case the fact is empty.
            if len(fact_words) == 0:
                score = 0.0
            else:
                overlap = fact_words & passage_words
                score = len(overlap) / len(fact_words)

            if score > best_score:
                best_score = score
                
        # Step 3: convert the real-valued score into a binary S/NS decision.
        # If enough of the fact's words were found in the best passage,
        # we treat the fact as supported. Otherwise, not supported.
        # The threshold itself is tuned on the dev set later (see sweep script).
        if best_score >= self.threshold:
            return "S"
        else:
            return "NS"
class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
