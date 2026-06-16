"""
RAG evaluation using RAGAS metrics.
Measures: faithfulness, answer relevancy, context recall.
"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall


def run_evaluation(questions: list, answers: list, contexts: list, ground_truths: list = None):
    """
    Evaluate RAG pipeline using RAGAS.

    Args:
        questions: list of input questions
        answers: list of generated answers
        contexts: list of lists of retrieved context strings
        ground_truths: optional list of reference answers

    Returns:
        dict of metric scores
    """
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)
    metrics = [faithfulness, answer_relevancy]
    if ground_truths:
        metrics.append(context_recall)

    result = evaluate(dataset, metrics=metrics)
    print("\nRAGAS Evaluation Results:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
    return result
