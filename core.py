import pandas as pd

from utils.logger import init_logger


class BaselinePipeline:
    def __init__(self):
        self.logger = init_logger()

    def process(self):
        """
            1. data_pipeline:
                    load dataset
            2. feature_extraction_pipeline: 
                    extract features and preprocess data
            3. model_pipeline: 
                    train model and estimate metric
        :return: exit code
        """
        try:
            processed = self.feature_extraction_pipeline()
            metric = self.model_pipeline()
        except ImportError as e:
            # TODO: handling ImportError
            self.logger.exception(e)
            return False
        except Exception as e:
            self.logger.exception(e)
            return False

        return True

    @property
    def dataset(self):
        return self.data_pipeline()

    @property
    def columns(self):
        return self.dataset.columns

    def data_pipeline(self):
        return self.load_dataset()

    def feature_extraction_pipeline(self):
        return 1

    def model_pipeline(self):
        return 1

    def load_dataset(self):
        return pd.read_excel(
            'origin/Concrete_Data.xls',
            sheet_name='Sheet1'
        )
