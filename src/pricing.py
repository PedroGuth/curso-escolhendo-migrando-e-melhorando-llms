
class PriceCalculator:
    """
    A class for calculating costs associated with AI model inference based on input and output tokens.
    
    Attributes:
        model_input_token_prices (dict): Dictionary of prices per 1000 input tokens by model ID
        model_output_token_prices (dict): Dictionary of prices per 1000 output tokens by model ID
    """
    
    def __init__(self):
        """Initialize the PriceCalculator with pricing data for various models."""
        # Dictionary containing prices per 1000 tokens for different models
        self.model_input_token_prices = {
            "amazon.nova-micro-v1:0": 0.000035,
            "amazon.nova-lite-v1:0": 0.00006,
            "amazon.nova-pro-v1:0": 0.0008,
            "us.anthropic.claude-3-5-haiku-20241022-v1:0": 0.0008,
            "anthropic.claude-3-sonnet-20240229-v1:0": 0.00300,
            "us.meta.llama3-2-11b-instruct-v1:0": 0.00016,
            "source_model": 0.00025
        }
        
        self.model_output_token_prices = {
            "amazon.nova-micro-v1:0": 0.00014,
            "amazon.nova-lite-v1:0": 0.00024,
            "amazon.nova-pro-v1:0": 0.0002,
            "us.anthropic.claude-3-5-haiku-20241022-v1:0": 0.004,
            "anthropic.claude-3-sonnet-20240229-v1:0": 0.015,
            "us.meta.llama3-2-11b-instruct-v1:0": 0.00016,
            "source_model": 0.003
        }
    
    def calculate_input_price(self, token_number, model_id):
        """
        Calculate the cost for a given number of input tokens based on the model used.
    
        Args:
            token_number (int): Number of input tokens.
            model_id (str): Identifier of the model.
            
        Returns:
            float: Cost calculated based on the input tokens and model used.
                  Returns 0 if model_id is not found.
        """
        if model_id in self.model_input_token_prices:
            price_per_1000_tokens = self.model_input_token_prices[model_id]
            cost = (token_number / 1000) * price_per_1000_tokens
            return round(cost, 8)
        else:
            return 0
    
    def calculate_output_price(self, token_number, model_id):
        """
        Calculate the cost for a given number of output tokens based on the model used.
    
        Args:
            token_number (int): Number of output tokens.
            model_id (str): Identifier of the model.
            
        Returns:
            float: Cost calculated based on the output tokens and model used.
                  Returns 0 if model_id is not found.
        """
        if model_id in self.model_output_token_prices:
            price_per_1000_tokens = self.model_output_token_prices[model_id]
            cost = (token_number / 1000) * price_per_1000_tokens
            return round(cost, 8)
        else:
            return 0
    
    def calculate_total_price(self, input_tokens, output_tokens, model_id):
        """
        Calculate comprehensive pricing information for model inference.
    
        Args:
            input_tokens (int): Number of input tokens.
            output_tokens (int): Number of output tokens.
            model_id (str): Identifier of the model.
            
        Returns:
            tuple: (input_cost, output_cost, total_cost, total_cost_1000)
                - input_cost: Cost for input tokens
                - output_cost: Cost for output tokens
                - total_cost: Total cost for one invocation
                - total_cost_1000: Total cost for 1000 invocations
        """
        # Calculate the input and output token costs
        input_cost = self.calculate_input_price(input_tokens, model_id)
        output_cost = self.calculate_output_price(output_tokens, model_id)
        
        # Calculate the total cost
        total_cost = round(input_cost + output_cost, 6)
        total_cost_1000 = round(total_cost * 1000, 6)
        
        return total_cost

    def calculate_llm_as_a_judge_evaluation_price(self, token_number, model_id):
        """
        Calculate the cost for a given number of input tokens based on the model used.
    
        Args:
            token_number (int): Number of input tokens.
            model_id (str): Identifier of the model.
            
        Returns:
            float: Cost calculated based on the input tokens and model used.
                  Returns 0 if model_id is not found.
        """
        if model_id in self.model_input_token_prices:
            price_per_1000_tokens = self.model_input_token_prices[model_id]
            cost = (token_number / 1000) * price_per_1000_tokens
            return round(cost, 8)
        else:
            return 0
    
    def get_supported_models(self):
        """
        Get a list of all supported model IDs.
        
        Returns:
            list: List of model IDs supported by the calculator
        """
        # Return the unique set of models from both price dictionaries
        return list(set(self.model_input_token_prices.keys()) | 
                   set(self.model_output_token_prices.keys()))
    
