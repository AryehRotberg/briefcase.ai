import random
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sentence_transformers import (SentenceTransformer,
                                   SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.training_args import BatchSamplers

from src.model_training.constants import Constants


class ModelTraining:
    """
    A class to handle the end-to-end process of training a SentenceTransformer model.
    """

    def __init__(
        self,
        pandas_dataframe_path: str | pd.DataFrame,
        model_checkpoint: str,
        seed: Optional[int] = 42,
    ) -> None:
        """
        Initializes the ModelTraining class.

        :param pandas_dataframe_path: Path to the CSV file containing the training data.
        :param model_checkpoint: Pretrained model checkpoint to use for fine-tuning.
        :param seed: Random seed for reproducibility.
        """
        if isinstance(pandas_dataframe_path, str):
            self.dataframe = pd.read_csv(pandas_dataframe_path)
        if isinstance(pandas_dataframe_path, pd.DataFrame):
            self.dataframe = pandas_dataframe_path

        self.corpus = list(self.dataframe["category"].unique())
        self.seed = seed
        self.model = SentenceTransformer(model_checkpoint)

        random.seed(self.seed)

    def create_triplets_from_data(
        self, num_to_sample_negatives: Optional[int] = 15
    ) -> List[Dict[str, str]]:
        """
        Creates triplets of anchor, positive, and negative samples from the dataframe.

        :param num_to_sample_negatives: Number of negative samples for each anchor.
        :return: List of triplets in the form of dictionaries.
        """
        triplets = []

        for _, row in self.dataframe.iterrows():
            anchor = row["source"]
            positive = row["category"]
            negatives = random.sample(
                [category for category in self.corpus if category != positive],
                num_to_sample_negatives,
            )

            for negative in negatives:
                triplets.append(
                    {"anchor": anchor, "positive": positive, "negative": negative}
                )

        return triplets

    def create_hf_dataset(
        self, triplets: List[Dict[str, str]], test_size: Optional[float] = 0.2
    ) -> Tuple[DatasetDict, DatasetDict]:
        """
        Converts triplets into a Hugging Face Dataset and splits it into train and test sets.

        :param triplets: List of triplets created from the data.
        :param test_size: Proportion of the dataset to include in the test split.
        :return: Train and test datasets as DatasetDict objects.
        """
        hf_dataset = Dataset.from_list(triplets)
        hf_dataset = hf_dataset.train_test_split(test_size=test_size, seed=self.seed)

        return hf_dataset["train"], hf_dataset["test"]

    def create_training_arguments(self, output_dir: str) -> SentenceTransformerTrainingArguments:
        """
        Creates training arguments for the SentenceTransformerTrainer.

        :param output_dir: Directory to save the trained model and checkpoints.
        :return: Training arguments object.
        """
        return SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=Constants.epochs,
            per_device_train_batch_size=Constants.train_batch_size,
            per_device_eval_batch_size=Constants.eval_batch_size,
            learning_rate=Constants.learning_rate,
            warmup_ratio=Constants.warmup_ratio,
            fp16=True,
            bf16=False,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy="steps",
            eval_steps=Constants.eval_steps,
            save_steps=100,
            save_total_limit=2,
            logging_steps=100,
        )

    def create_triplet_evaluator(self, eval_dataset: DatasetDict) -> TripletEvaluator:
        """
        Creates a triplet evaluator for evaluation during training.

        :param eval_dataset: Evaluation dataset.
        :return: TripletEvaluator object.
        """
        return TripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            name="all-nli-dev"
        )

    def create_trainer(
        self,
        train_dataset: DatasetDict,
        eval_dataset: DatasetDict,
        training_args: SentenceTransformerTrainingArguments,
        loss_function,
        evaluator: TripletEvaluator,
    ) -> SentenceTransformerTrainer:
        """
        Creates a SentenceTransformerTrainer for model training.

        :param train_dataset: Training dataset.
        :param eval_dataset: Evaluation dataset.
        :param training_args: Training arguments.
        :param loss_function: Loss function to use for training.
        :param evaluator: Evaluator for validation during training.
        :return: SentenceTransformerTrainer object.
        """
        return SentenceTransformerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss_function(self.model),
            evaluator=evaluator,
        )

    def train(
        self,
        train_dataset: DatasetDict,
        eval_dataset: DatasetDict,
        loss_function,
        fine_tuned_model_output_dir: str,
    ) -> SentenceTransformerTrainer:
        """
        Trains the SentenceTransformer model.

        :param train_dataset: Training dataset.
        :param eval_dataset: Evaluation dataset.
        :param loss_function: Loss function to use for training.
        :param fine_tuned_model_output_dir: Directory to save the fine-tuned model.
        :return: Trained SentenceTransformerTrainer object.
        """
        evaluator = self.create_triplet_evaluator(eval_dataset)
        training_args = self.create_training_arguments(fine_tuned_model_output_dir)
        trainer = self.create_trainer(
            train_dataset, eval_dataset, training_args, loss_function, evaluator
        )

        print("Evaluating cosine accuracy of the data...")
        print(evaluator(self.model))
        print("Starting training...")
        trainer.train()

        return trainer

    def save_model(self, trainer: SentenceTransformerTrainer, path: str) -> None:
        """
        Saves the fine-tuned model to the specified path.

        :param trainer: Trained SentenceTransformerTrainer object.
        :param path: Path to save the fine-tuned model.
        """
        trainer.model.save_pretrained(path)
