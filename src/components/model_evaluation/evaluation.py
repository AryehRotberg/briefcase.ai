from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm


class ModelEvaluation:
    def __init__(self, dataframe: pd.DataFrame, model_path: str):
        """
        Initialize the ModelEvaluation class.

        :param dataframe: DataFrame containing the data to evaluate.
        :param model_path: Path to the pre-trained model.
        """
        self.dataframe = dataframe
        self.model = SentenceTransformer(model_path)
    
    def _predict(self) -> List[str]:
        """
        Selects the best chunks based on cosine similarity.
        
        :return: List of predicted categories.
        """
        results = []
        categories = list(self.dataframe["category"].unique())
        category_embeddings = self.model.encode(categories, convert_to_tensor=True)

        for query in tqdm(self.dataframe["source"], total=len(self.dataframe["source"])):
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            cosine_scores = util.cos_sim(query_embedding, category_embeddings)[0]
            results.append(categories[torch.argmax(cosine_scores).item()])
        
        return results
    
    def evaluate(self) -> Tuple[float, np.ndarray, Dict[str, Dict[str, float]]]:
        """
        Evaluates the model using F1 score and confusion matrix.

        :return: F1 score, confusion matrix, and classification report.
        """
        y_true = self.dataframe["category"].tolist()
        y_pred = self._predict()
        labels = sorted(list(set(y_true) | set(y_pred)))

        f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

        print(f"F1 Score (Weighted Average): {f1_weighted:.4f}")
        print(f"F1 Score (Macro Average): {f1_macro:.4f}\n")

        report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose()

        class_metrics_df = report_df.loc[labels]
        overall_metrics_df = report_df.drop(labels)

        sorted_class_metrics_df = class_metrics_df.sort_values(by="f1-score", ascending=False)

        print("Classification Report (Sorted by F1-score Descending):")

        print(sorted_class_metrics_df.to_string(float_format='{:.4f}'.format))
        print("-" * 60)
        print(overall_metrics_df.to_string(float_format='{:.4f}'.format))

        print("\nDisplaying Confusion Matrix...")
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        return f1_weighted, cm, report_dict
    
    def predict_query(self, query: str, top_k: int=3) -> None:
        """
        Predicts the top K most similar sentences in the corpus for a given query.

        :param query: The query string to search for.
        :param top_k: The number of top results to return.
        """
        categories = list(self.dataframe["category"].unique())

        category_embeddings = self.model.encode(categories, convert_to_tensor=True)
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        cosine_scores = util.cos_sim(query_embedding, category_embeddings)[0]
        top_results = torch.topk(cosine_scores, k=top_k)

        print(f"\nQuery: {query}\n")
        print(f"Top {top_k} most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            print(f"- Score: {score:.4f}")
            print(f"  Text: {categories[idx]}")
            print("-" * 50)
