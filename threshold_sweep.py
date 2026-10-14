"""
Threshold sweep for the word overlap classifier.

Runs WordRecallThresholdFactChecker across a range of thresholds on the
labelled dev set and prints the accuracy at each. The best threshold found
here is what we hard-code (via the default) in factcheck.py for submission.

This script is NOT required for the submission - it only produces the
numbers that go into the methodology/results section of the report.

Run with:  python threshold_sweep.py
"""

from factcheck import WordRecallThresholdFactChecker
from factchecking_main import read_passages, read_fact_examples

def evaluate(fact_checker, examples):
    # Runs the given fact_checker over every example (skipping IR as the main
    # driver does) and returns accuracy as a float in [0, 1]. Kept separate
    # from the sweep loop so we can reuse it for other experiments later
    # (preprocessing ablation, alternative scorers).
    correct = 0
    total = 0
    for ex in examples:
        # The brief says IR (irrelevant) labels are ignored during scoring,
        # so we match that convention — only S and NS examples count.
        if ex.label not in ("S", "NS"):
            continue
        prediction = fact_checker.predict(ex.fact, ex.passages)
        if prediction == ex.label:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def main():
    # Load the data once using the lecturer's own loaders so we match the
    # exact behaviour of factchecking_main.py. Doing this once at the top
    # (rather than inside the loop) saves ~2 seconds per threshold.
    print("Loading data...")
    fact_to_passage_dict = read_passages("data/passages_bm25_ChatGPT_humfacts.jsonl")
    examples = read_fact_examples("data/dev_labeled_ChatGPT.jsonl", fact_to_passage_dict)
    print(f"Loaded {len(examples)} examples")

    # Sweep thresholds from 0.0 to 1.0 in steps of 0.05. 21 points gives a
    # fine enough grid to find the real optimum without being so dense that
    # the output becomes noise. We can narrow in later if the peak looks
    # ambiguous.
    print("\nThreshold sweep:")
    print(f"{'threshold':>10}  {'accuracy':>8}")
    print("-" * 22)

    best_threshold = 0.0
    best_accuracy = 0.0

    # arange would be nicer but we'd need numpy; a hand-rolled loop is clear
    # enough and avoids introducing a dependency for a single number.
    threshold = 0.0
    while threshold <= 1.0 + 1e-9:
        # Build a fresh classifier at this threshold. Keep all other
        # preprocessing options at their defaults (NLTK + stopwords + stemming)
        # since Commit 2 showed that's our best configuration so far.
        fact_checker = WordRecallThresholdFactChecker(threshold=threshold)
        accuracy = evaluate(fact_checker, examples)

        print(f"{threshold:>10.2f}  {accuracy:>8.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

        threshold += 0.05

    print("-" * 22)
    print(f"\nBest threshold: {best_threshold:.2f}")
    print(f"Best accuracy:  {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")


if __name__ == "__main__":
    main()
