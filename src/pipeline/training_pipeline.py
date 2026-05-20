"""
Training Pipeline
=================
Complete end-to-end training workflow orchestration.
Coordinates all components to train and evaluate models.
"""

import pandas as pd
from typing import Dict, Tuple, Optional
from src.logger import get_logger
from src.exceptions import NLPException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.feature_engineering import FeatureEngineering
from src.components.model_trainer import ModelTrainer, NERTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.summarizer import TextSummarizer


logger = get_logger(__name__)


class TrainingPipeline:
    """
    Orchestrates complete training pipeline for text classification.
    
    Pipeline stages:
    1. Data Ingestion: Load and validate dataset
    2. Data Transformation: Preprocess and clean text
    3. Feature Engineering: Extract TF-IDF features
    4. Model Training: Train classification model
    5. Model Evaluation: Evaluate and save metrics
    """
    
    def __init__(self):
        """Initialize training pipeline."""
        self.data_ingestion = None
        self.data_transformation = None
        self.feature_engineering = None
        self.model_trainer = None
        self.model_evaluation = None
        
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        logger.info("TrainingPipeline initialized")
    
    
    def run(
        self,
        dataset_path: str,
        test_size: float = 0.2,
        remove_stopwords: bool = True,
        save_artifacts: bool = True
    ) -> Dict:
        """
        Execute complete training pipeline.
        
        Args:
            dataset_path (str): Path to dataset CSV
            test_size (float): Proportion of data for testing
            remove_stopwords (bool): Whether to remove stopwords during preprocessing
            save_artifacts (bool): Whether to save trained models and artifacts
            
        Returns:
            dict: Pipeline execution results
        """
        try:
            logger.info("="*60)
            logger.info("Starting Training Pipeline")
            logger.info("="*60)
            
            # 1. Data Ingestion
            logger.info("\n[STAGE 1] Data Ingestion")
            self._data_ingestion_stage(dataset_path)
            
            # 2. Data Transformation
            logger.info("\n[STAGE 2] Data Transformation")
            self._data_transformation_stage(remove_stopwords)
            
            # 3. Data Split
            logger.info("\n[STAGE 3] Train-Test Split")
            self._data_split_stage(test_size)
            
            # 4. Feature Engineering
            logger.info("\n[STAGE 4] Feature Engineering")
            self._feature_engineering_stage()
            
            # 5. Model Training
            logger.info("\n[STAGE 5] Model Training")
            self._model_training_stage()
            
            # 6. Model Evaluation
            logger.info("\n[STAGE 6] Model Evaluation")
            evaluation_results = self._model_evaluation_stage()
            
            # 7. Save Artifacts
            if save_artifacts:
                logger.info("\n[STAGE 7] Saving Artifacts")
                self._save_artifacts_stage()
            
            logger.info("\n" + "="*60)
            logger.info("Training Pipeline Completed Successfully!")
            logger.info("="*60)
            
            return {
                "status": "success",
                "evaluation_results": evaluation_results,
                "data_stats": self._get_pipeline_stats()
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    
    def _data_ingestion_stage(self, dataset_path: str) -> None:
        """Execute data ingestion stage."""
        try:
            self.data_ingestion = DataIngestion(dataset_path)
            self.data_ingestion.load_data()
            self.data_ingestion.validate_data()
            
            stats = self.data_ingestion.get_statistics()
            logger.info(f"Dataset shape: {stats['shape']}")
            logger.info(f"Classes: {list(stats['label_distribution'].keys())}")
            
            self.data_ingestion.save_ingestion_report()
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {str(e)}")
            raise
    
    
    def _data_transformation_stage(self, remove_stopwords: bool) -> None:
        """Execute data transformation stage."""
        try:
            self.data_transformation = DataTransformation(
                remove_stopwords=remove_stopwords
            )
            
            raw_data = self.data_ingestion.get_data()
            self.train_df = self.data_transformation.transform_dataframe(raw_data)
            
            logger.info(f"Transformation complete: {len(self.train_df)} samples")
            
        except Exception as e:
            logger.error(f"Data transformation failed: {str(e)}")
            raise
    
    
    def _data_split_stage(self, test_size: float) -> None:
        """Execute train-test split stage."""
        try:
            self.train_df, self.test_df = (
                self.data_transformation.split_train_test(
                    self.train_df,
                    test_size=test_size
                )
            )
            
        except Exception as e:
            logger.error(f"Data split failed: {str(e)}")
            raise
    
    
    def _feature_engineering_stage(self) -> None:
        """Execute feature engineering stage."""
        try:
            self.feature_engineering = FeatureEngineering()
            
            # Fit on training data, transform both train and test
            self.X_train = self.feature_engineering.fit_transform(
                self.train_df['text_cleaned'].tolist()
            )
            self.X_test = self.feature_engineering.transform(
                self.test_df['text_cleaned'].tolist()
            )
            
            self.y_train = self.train_df['label'].values
            self.y_test = self.test_df['label'].values
            
            logger.info(f"Feature matrix shapes - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise
    
    
    def _model_training_stage(self) -> None:
        """Execute model training stage."""
        try:
            self.model_trainer = ModelTrainer(model_type="logistic_regression")
            self.model_trainer.train(self.X_train, self.y_train)
            
            logger.info("Model trained successfully")
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    
    def _model_evaluation_stage(self) -> Dict:
        """Execute model evaluation stage."""
        try:
            self.model_evaluation = ModelEvaluation()
            
            # Make predictions
            y_pred = self.model_trainer.predict(self.X_test)
            y_proba = self.model_trainer.predict_proba(self.X_test)
            
            # Compute metrics
            metrics = self.model_evaluation.compute_metrics(
                self.y_test,
                y_pred,
                y_proba=y_proba
            )
            
            self.model_evaluation.print_metrics()
            self.model_evaluation.save_metrics()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    
    def _save_artifacts_stage(self) -> None:
        """Save trained models and artifacts."""
        try:
            # Save classification model
            model_path = self.model_trainer.save_model()
            logger.info(f"Classification model saved: {model_path}")
            
            # Save vectorizer
            vectorizer_path = self.feature_engineering.vectorizer
            import joblib
            joblib.dump(
                vectorizer_path,
                f"models/tfidf_vectorizer.pkl"
            )
            logger.info("TF-IDF vectorizer saved")
            
        except Exception as e:
            logger.error(f"Artifact saving failed: {str(e)}")
            raise
    
    
    def _get_pipeline_stats(self) -> Dict:
        """Get summary statistics of pipeline."""
        return {
            "train_samples": len(self.train_df) if self.train_df is not None else 0,
            "test_samples": len(self.test_df) if self.test_df is not None else 0,
            "feature_count": self.X_train.shape[1] if self.X_train is not None else 0,
        }


class NLPInferencePipeline:
    """
    Complete inference pipeline combining all NLP tasks.
    
    Includes:
    - Text Classification
    - Named Entity Recognition
    - Text Summarization
    
    Extracted from: 06_INFERENCE.ipynb
    """
    
    def __init__(
        self,
        classifier_path: str = "models/text_classifier.pkl",
        vectorizer_path: str = "models/tfidf_vectorizer.pkl",
        ner_model: str = "en_core_web_sm",
        summarizer_model: str = "t5-small"
    ):
        """
        Initialize inference pipeline.
        
        Args:
            classifier_path (str): Path to trained classifier
            vectorizer_path (str): Path to TF-IDF vectorizer
            ner_model (str): spaCy NER model name
            summarizer_model (str): T5 model name for summarization
        """
        import joblib
        import spacy
        
        try:
            logger.info("Loading inference pipeline components...")
            
            # Load classifier and vectorizer
            self.classifier = joblib.load(classifier_path)
            self.vectorizer = joblib.load(vectorizer_path)
            
            # Load NER
            self.nlp = spacy.load(ner_model)
            
            # Load summarizer
            self.summarizer = TextSummarizer(summarizer_model)
            
            logger.info("Inference pipeline loaded successfully")
            
        except Exception as e:
            raise NLPException(f"Failed to load inference pipeline: {str(e)}")
    
    
    def process_document(self, text: str) -> Dict:
        """
        Process document through complete NLP pipeline.
        
        Args:
            text (str): Input document text
            
        Returns:
            dict: Results from classification, NER, and summarization
        """
        try:
            if not text or len(text.strip()) < 20:
                raise NLPException("Input text too short (min 20 characters)")
            
            logger.info(f"Processing document ({len(text)} chars)")
            
            # 1. Classification
            X = self.vectorizer.transform([text])
            category = self.classifier.predict(X)[0]
            
            # 2. NER
            doc = self.nlp(text)
            entities = [
                {
                    "text": ent.text,
                    "label": ent.label_
                }
                for ent in doc.ents
            ]
            
            # 3. Summarization
            summary = self.summarizer.summarize(text)
            
            result = {
                "category": category,
                "entities": entities,
                "summary": summary,
                "text_length": len(text),
                "entity_count": len(entities)
            }
            
            logger.info(f"Document processing complete. Category: {category}")
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise
