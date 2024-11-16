from transformers import pipeline, set_seed
import logging

# Configure logging for better error tracking
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    # Initialize the model pipeline for text generation
    logger.info("Loading model pipeline...")
    chatbot_pipeline = pipeline('text-generation', model='microsoft/DialoGPT-medium')
    logger.info("Model pipeline loaded successfully.")
    
    # Optional: Set a seed for reproducible results
    set_seed(42)

except Exception as e:
    logger.error(f"Error loading the model pipeline: {str(e)}")
    chatbot_pipeline = None  # Ensure we handle this in the get_response function

# Function to generate chatbot responses
def get_response(user_input):
    if chatbot_pipeline is None:
        logger.error("Model pipeline is not initialized.")
        return "An error occurred with the chatbot model initialization."

    try:
        # Generate a response from the model
        response = chatbot_pipeline(user_input, max_length=100, pad_token_id=50256)
        return response[0]['generated_text']
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "An error occurred while generating a response. Please try again."
