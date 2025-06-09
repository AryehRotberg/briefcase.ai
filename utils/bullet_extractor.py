from collections import defaultdict

import nltk
import torch
from sentence_transformers import util

from utils.categories import Categories


class BulletExtractor:
    def __init__(self, model, category_embeddings, corpus: list, threshold: float = 0.65, window_sizes: list = [1, 2, 3, 4, 5]):
        """
        Initializes the BulletExtractor with the necessary components.
        """
        self.model = model
        self.category_embeddings = category_embeddings
        self.corpus = corpus
        self.threshold = threshold
        self.window_sizes = window_sizes

    def extract(self, doc: str) -> list:
        """
        Extracts key bullet points from the document.
        """
        sentences = nltk.sent_tokenize(doc)

        chunk_texts, chunk_meta = self._prepare_chunks(sentences)
        chunk_embeddings = self._encode_chunks(chunk_texts)
        best_chunks = self._select_best_chunks(chunk_embeddings, chunk_meta, sentences)
        filtered_chunks = self._filter_chunks(best_chunks)

        return self._categorize_chunks(filtered_chunks)

    def _prepare_chunks(self, sentences: list) -> tuple:
        """
        Generates all possible chunks from sentences with corresponding metadata.
        """
        chunk_texts = []
        chunk_meta = []

        for i in range(len(sentences)):
            for window_size in self.window_sizes:
                chunk = self._get_chunk(sentences, i, window_size)
                chunk_texts.append(chunk)
                chunk_meta.append((i, window_size))

        return chunk_texts, chunk_meta

    def _encode_chunks(self, chunk_texts: list):
        """
        Encodes all chunks using the embedding model.
        """
        return self.model.encode(
            chunk_texts,
            convert_to_tensor=True,
            batch_size=128,
            show_progress_bar=False
        )

    def _select_best_chunks(self, chunk_embeddings, chunk_meta, sentences: list) -> dict:
        """
        Selects the best scoring chunk per sentence based on category similarity.
        """
        best_chunks = {}

        for emb, (i, window_size) in zip(chunk_embeddings, chunk_meta):
            score, category = self._predict_embedding(emb)

            if (i not in best_chunks) or (score > best_chunks[i]["score"]):
                start = max(0, i - (window_size - 1) // 2)
                end = min(len(sentences), start + window_size)

                best_chunks[i] = {
                    "i": i,
                    "window": window_size,
                    "score": score,
                    "category": category,
                    "span": (start, end),
                    "text": " ".join(sentences[start:end])
                }

        return best_chunks

    def _filter_chunks(self, best_chunks: dict) -> list:
        """
        Filters chunks based on score threshold and overlap.
        """
        selected = []
        used_indices = set()

        for candidate in sorted(best_chunks.values(), key=lambda x: x["score"], reverse=True):
            if candidate["score"] < self.threshold:
                continue

            span_indices = range(candidate["span"][0], candidate["span"][1])
            if any(idx in used_indices for idx in span_indices):
                continue

            selected.append(candidate)
            used_indices.update(span_indices)

        return selected
    
    def _categorize_chunks(self, filtered_chunks: dict) -> dict:
        categorized_bullets = defaultdict(set)
        sorted_criticality_dict = {}

        for bullet in filtered_chunks:
            categorized_bullets[bullet["category"]].add(bullet["text"])
        
        for crit_list in (Categories.high_criticality,
                          Categories.medium_criticality,
                          Categories.low_criticality):
            for category in crit_list:
                if category in categorized_bullets:
                    sorted_criticality_dict[category] = categorized_bullets[category]
        
        return sorted_criticality_dict

    def _get_chunk(self, sentences: list, center: int, window_size: int) -> str:
        """
        Extracts a chunk of sentences centered around a specific index.
        """
        start = max(0, center - (window_size - 1) // 2)
        end = min(len(sentences), start + window_size)
        return " ".join(sentences[start:end])

    def _predict_embedding(self, query_emb) -> tuple:
        """
        Predicts the best matching category for a given query embedding.
        """
        cosine_scores = util.cos_sim(query_emb, self.category_embeddings)[0]
        idx = torch.argmax(cosine_scores).item()
        return cosine_scores[idx].item(), self.corpus[idx]
