# models/model.py

"""
Model Definition Module
"""

from transformers import GPT2LMHeadModel

def get_model(model_name):
    """
    Initialize the model.

    Args:
        model_name (str): Name of the pre-trained model.

    Returns:
        model: The pre-trained language model.
    """
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model
