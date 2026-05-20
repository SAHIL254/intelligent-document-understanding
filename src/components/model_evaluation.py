"""
Model Evaluation Component
===========================
Handles model evaluation and performance metrics.
Extracted from: 03_text_classification.ipynb
"""

import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from src.logger import get_logger
from src.exceptions import ModelEvaluationError
from src.utils import ARTIFACTS_DIR
import json


logger = get_logger(__name__)


class ModelEvaluation:
    """
    Evaluates trained classification models and computes metrics.
    """
    
    def __init__(self):
        """Initialize model evaluation."""
        self.metrics = None
        logger.info("ModelEvaluation initialized")
    
    
    def compute_metrics(
        self,
        y_true,
        y_pred,
        y_proba=None,
        average: str = 'weighted'
    ) -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional, for ROC-AUC)
            average (str): Averaging method ('weighted', 'macro', 'micro')
            
        Returns:
            dict: Dictionary containing all metrics
            
        Raises:
            ModelEvaluationError: If metric computation fails
        """
        try:
            logger.info("Computing evaluation metrics")
            
            self.metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
            }
            
            # Add per-class metrics
            try:
                self.metrics["classification_report"] = classification_report(
                    y_true, y_pred, output_dict=True
                )
            except Exception as e:
                logger.warning(f"Could not compute classification report: {str(e)}")
            
            # Add confusion matrix
            try:
                cm = confusion_matrix(y_true, y_pred)
                self.metrics["confusion_matrix"] = cm.tolist()
            except Exception as e:
                logger.warning(f"Could not compute confusion matrix: {str(e)}")
            
            # Add ROC-AUC if probabilities provided (only for binary classification)
            if y_proba is not None:
                try:
                    if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                        self.metrics["roc_auc"] = float(roc_auc)
                except Exception as e:
                    logger.warning(f"Could not compute ROC-AUC: {str(e)}")
            
            logger.info(f"Metrics computed: Accuracy={self.metrics['accuracy']:.4f}, "
                       f"F1={self.metrics['f1']:.4f}")
            
            return self.metrics
            
        except Exception as e:
            raise ModelEvaluationError(f"Metric computation failed: {str(e)}")
    
    
    def get_metrics_summary(self) -> str:
        """
        Get a formatted summary of metrics.
        
        Returns:
            str: Formatted metrics summary
            
        Raises:
            ModelEvaluationError: If no metrics computed yet
        """
        if self.metrics is None:
            raise ModelEvaluationError("No metrics computed yet. Call compute_metrics() first.")
        
        summary = (
            f"Model Evaluation Summary\n"
            f"{'='*50}\n"
            f"Accuracy:  {self.metrics['accuracy']:.4f}\n"
            f"Precision: {self.metrics['precision']:.4f}\n"
            f"Recall:    {self.metrics['recall']:.4f}\n"
            f"F1-Score:  {self.metrics['f1']:.4f}\n"
        )
        
        if 'roc_auc' in self.metrics:
            summary += f"ROC-AUC:   {self.metrics['roc_auc']:.4f}\n"
        
        summary += f"{'='*50}"
        
        return summary
    
    
    def save_metrics(self, output_path: str = None) -> str:
        """
        Save metrics to JSON file.
        
        Args:
            output_path (str, optional): Path to save metrics
            
        Returns:
            str: Path where metrics were saved
            
        Raises:
            ModelEvaluationError: If metrics not computed or saving fails
        """
        try:
            if self.metrics is None:
                raise ModelEvaluationError("No metrics computed yet. Call compute_metrics() first.")
            
            if output_path is None:
                output_path = str(ARTIFACTS_DIR / "evaluation_metrics.json")
            
            # Handle non-serializable types
            metrics_to_save = self.metrics.copy()
            
            # Convert numpy arrays in classification report to lists
            if "classification_report" in metrics_to_save:
                for key, value in metrics_to_save["classification_report"].items():
                    if isinstance(value, dict):
                        for metric_key, metric_value in value.items():
                            if hasattr(metric_value, 'item'):
                                value[metric_key] = float(metric_value)
            
            with open(output_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=4)
            
            logger.info(f"Metrics saved to {output_path}")
            return output_path
            
        except Exception as e:
            raise ModelEvaluationError(f"Failed to save metrics: {str(e)}")
    
    
    def print_metrics(self) -> None:
        """
        Print formatted metrics to console.
        
        Raises:
            ModelEvaluationError: If no metrics computed yet
        """
        if self.metrics is None:
            raise ModelEvaluationError("No metrics computed yet. Call compute_metrics() first.")
        
        print(self.get_metrics_summary())
    
    
    @staticmethod
    def evaluate_model(
        model,
        X_test,
        y_test,
        feature_names=None,
        save_report: bool = False
    ) -> Dict:
        """
        Complete evaluation of a model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            feature_names: Optional feature names for interpretation
            save_report: Whether to save evaluation report
            
        Returns:
            dict: Evaluation results
        """
        try:
            logger.info("Starting model evaluation")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Try to get probabilities
            y_proba = None
            try:
                y_proba = model.predict_proba(X_test)
            except:
                pass
            
            # Compute metrics
            evaluator = ModelEvaluation()
            metrics = evaluator.compute_metrics(y_test, y_pred, y_proba)
            
            if save_report:
                evaluator.save_metrics()
            
            evaluator.print_metrics()
            
            return metrics
            
        except Exception as e:
            raise ModelEvaluationError(f"Model evaluation failed: {str(e)}")


class NEREvaluation:
    """
    Evaluation metrics for Named Entity Recognition.
    """
    
    @staticmethod
    def evaluate_ner_extraction(
        extracted_entities: list,
        ground_truth_entities: list
    ) -> Dict:
        """
        Evaluate NER extraction quality.
        
        Args:
            extracted_entities (list): Entities extracted by model
            ground_truth_entities (list): Ground truth entities
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            # Simple exact match evaluation
            extracted_set = set(
                (ent['text'], ent['label']) for ent in extracted_entities
            )
            ground_set = set(
                (ent['text'], ent['label']) for ent in ground_truth_entities
            )
            
            true_positives = len(extracted_set & ground_set)
            false_positives = len(extracted_set - ground_set)
            false_negatives = len(ground_set - extracted_set)
            
            precision = true_positives / (true_positives + false_positives) \
                if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) \
                if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) \
                if (precision + recall) > 0 else 0
            
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            }
            
        except Exception as e:
            raise ModelEvaluationError(f"NER evaluation failed: {str(e)}")
