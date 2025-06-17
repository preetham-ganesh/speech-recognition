import os
import sys
import warnings
import logging
import argparse


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.FATAL)


class SpeechRecognition(object):
    """"""

    def __init__(self, model_version: str) -> None:
        """Creates object attributes for the SpeechRecognition class.

        Creates object attributes for the SpeechRecognition class.

        Args:
            model_version: A string for the version of the model should be used for prediction.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(model_version, str), "Variable model_version of type 'str'."

        # Initalizes class variables.
        self.model_version = model_version
