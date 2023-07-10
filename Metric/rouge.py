from rouge_score import rouge_scorer

class Metric:

    def rouge(self, hypo, ref):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        return scorer.score(hypo,ref)
