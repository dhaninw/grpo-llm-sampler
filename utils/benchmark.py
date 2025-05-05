import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import re
import numpy as np
from fractions import Fraction

#################################
######## Response Parser ########
#################################

def extract_boxed_content(text):
    """
    Extract content inside \boxed{} with proper handling of nested braces.
    """
    boxed_pattern = '\\boxed{'
    
    # Find the starting position of \boxed{
    start_pos = text.find(boxed_pattern)
    if start_pos == -1:
        boxed_pattern = r'\\boxed{'
        start_pos = text.find(boxed_pattern)
        if start_pos == -1:
            return None
    
    # Move to the position after the opening brace
    pos = start_pos + len(boxed_pattern)
    brace_count = 1  # We've already seen one opening brace
    start_content = pos
    
    # Iterate through the string to find the matching closing brace
    while pos < len(text) and brace_count > 0:
        if text[pos] == '{':
            brace_count += 1
        elif text[pos] == '}':
            brace_count -= 1
        pos += 1
    
    if brace_count == 0:  # Found matching closing brace
        return text[start_content:pos-1]  # Extract content without the enclosing braces
    else:
        return None  # No matching closing brace found

def extract_answer(response):
    # Extract the answer from the response using regex
    answer = extract_boxed_content(response)
    if answer:
        answer = answer.strip()
    if answer is None:
        temp = r'<answer>.+</answer>'
        m = re.search(temp, response)
        if m:
            answer = m.group(1).strip()
    if answer is None:
        lines = response.strip().split('\n')
        if lines:
            # Assume the last line contains the final answer (could use regex to pull numbers)
            ans = lines[-1].strip()
        else:
            ans = response.strip()
        try:
            return np.around(latex_to_float(ans), 5)
        except:
            return ans
    
    result = latex_to_float(answer)
    if isinstance(result, float):
        return np.around(result, 5)
    else:
        return result
    
def latex_to_float(latex_expr):
    """
    Convert LaTeX mathematical expressions to float values.
    Handles fractions, square roots, exponents, and basic operations.
    """
    # Remove any whitespace and $ symbols
    latex_expr = latex_expr.strip().replace('$', '').replace(' ', '')
    
    # Handle fractions: \frac{numerator}{denominator}
    frac_pattern = r'\\frac\{([^{}]+)\}\{([^{}]+)\}'
    while re.search(frac_pattern, latex_expr):
        match = re.search(frac_pattern, latex_expr)
        if match:
            num = latex_to_float(match.group(1))
            denom = latex_to_float(match.group(2))
            latex_expr = latex_expr.replace(match.group(0), str(num / denom))
    
    # Handle fractions: \dfrac{numerator}{denominator}
    frac_pattern = r'\\dfrac\{([^{}]+)\}\{([^{}]+)\}'
    while re.search(frac_pattern, latex_expr):
        match = re.search(frac_pattern, latex_expr)
        if match:
            num = latex_to_float(match.group(1))
            denom = latex_to_float(match.group(2))
            latex_expr = latex_expr.replace(match.group(0), str(num / denom))

    textbf_pattern = r'\\textbf\{([^{}]+)\}'
    while re.search(textbf_pattern, latex_expr):
        match = re.search(textbf_pattern, latex_expr)
        if match:
            expr = latex_to_float(match.group(1))
            latex_expr = latex_expr.replace(match.group(0), str(expr))
    
    # Handle square roots: \sqrt{expression}
    sqrt_pattern = r'\\sqrt\{([^{}]+)\}'
    while re.search(sqrt_pattern, latex_expr):
        match = re.search(sqrt_pattern, latex_expr)
        if match:
            expr = latex_to_float(match.group(1))
            latex_expr = latex_expr.replace(match.group(0), str(expr ** 0.5))
    
    # Handle powers: {base}^{exponent} or base^{exponent}
    power_pattern1 = r'\{([^{}]+)\}\^\{([^{}]+)\}'
    power_pattern2 = r'([0-9.-]+)\^\{([^{}]+)\}'
    
    # Process {base}^{exponent}
    while re.search(power_pattern1, latex_expr):
        match = re.search(power_pattern1, latex_expr)
        if match:
            base = latex_to_float(match.group(1))
            exp = latex_to_float(match.group(2))
            latex_expr = latex_expr.replace(match.group(0), str(base ** exp))
    
    # Process base^{exponent}
    while re.search(power_pattern2, latex_expr):
        match = re.search(power_pattern2, latex_expr)
        if match:
            base = float(match.group(1))
            exp = latex_to_float(match.group(2))
            latex_expr = latex_expr.replace(match.group(0), str(base ** exp))
    
    # Handle basic negation: handle cases like -\frac{...}
    if latex_expr.startswith('-') and '/' in latex_expr[1:]:
        return -float(Fraction(latex_expr[1:]))
    
    # Try to evaluate as a fraction (handles cases like -5/12)
    if '/' in latex_expr:
        return float(Fraction(latex_expr))
    
    # Handle specific LaTeX constants
    if latex_expr == '\\pi':
        return 3.141592653589793
    elif latex_expr == '\\e':
        return 2.718281828459045
    
    # Try to convert to float directly
    try:
        return float(latex_expr.replace('\\', ''))
    except ValueError:
        # If we can't convert directly, try to evaluate as a Python expression
        try:
            return eval(latex_expr)
        except:
            # If all else fails, return the original expression
            return latex_expr
# latex_to_float(r'$\frac{4}{5}')


##################################
############# Models #############
##################################
def oracle_model(prompt, num_samples=1):
    """
    Generate answers using the model with the given prompt.
    """
    for i in range(len(aime_df)):
        if aime_df.iloc[i]['Problem'] in prompt:
            # Use the solution as the answer
            return [aime_df.iloc[i]['Solution']] * num_samples
    return ["33"] * num_samples

def foo_model(prompt, num_samples=1):
    return ["33"] * num_samples

from vllm import SamplingParams

class FastMathModel:
    def __init__(self, model, tokenizer, sampling_params=None, max_tokens=20000):
        self.model = model
        self.tokenizer = tokenizer
        if sampling_params is None:
            sampling_params = SamplingParams(
            temperature = 1.0,
            top_p = 0.9,
            max_tokens = max_tokens)
        self.sampling_params = sampling_params
    
    def generate(self, prompt, num_samples=1):
        # Tokenize the input prompt
        if num_samples != 1:
            self.sampling_params.n = num_samples
        text = self.tokenizer.apply_chat_template([
            {"role" : "user", "content" : prompt},
        ], tokenize = False, add_generation_prompt = True)
        results = self.model.fast_generate(
            [text],
            sampling_params = self.sampling_params,
            lora_request = None,
        )
        return [result.outputs[0].text for result in results]
   
from vllm.logits_process import LogitsProcessor

class LessThoughtSwitchModel(FastMathModel):
    def __init__(self, model, tokenizer, sampling_params=None, max_tokens=20000):
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature = 0.6,
                repetition_penalty = 1.1,
                top_p = 0.9,
                max_tokens = max_tokens,
                logits_processors = [self._penalize_thought_switch]
            )
        super().__init__(model, tokenizer, sampling_params)
        self.penalized_tokens = self._thought_switch_penalty()
        if sampling_params.logits_processors is None:
            sampling_params.logits_processors = [self._penalize_thought_switch]
        self.sampling_params = sampling_params
        

    def _thought_switch_penalty(self):
        custom_penalties = {
            "Wait": 3.0,  # Strong penalty for "Wait"
            "wait": 3.0,
            "Hmm": 3.0,   # Strong penalty for hesitation markers
            "hmm": 3.0
        }
        # Get token IDs for penalized tokens
        penalized_tokens = {}
        for word, penalty in custom_penalties.items():
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            for token_id in token_ids:
                penalized_tokens[token_id] = penalty
        return penalized_tokens

    def _penalize_thought_switch(self, token_ids, logits):
        for token_id, penalty in self.penalized_tokens.items():
            logits[token_id] -= penalty
        return logits

####################################
######### Model Comparison #########
####################################

def get_prompt_template():
    system_prompt = r'''Assume you are a professional mathematician.{}
    Please show your step by step reasoning in <think> tag. keep your 
    solution concise skipping arithmetic details, and generate your answer as 
    a number in <answer> tag.
    Constraint: Please keep your response in the following format:
    <think>
    [your reasoning]
    </think>
    [your derivation, keep this short] 
    <answer>
    [succinct answer]
    </answer>
    '''
    return system_prompt

def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def best_answer_accuracy(model_answers, truth):
    """
    Calculate the accuracy of the best of model's answers against the ground truth.
    """
    # Convert all answers to float for comparison
    correct_answers = []
    for ans in model_answers:
        try:
            cor = float(ans == truth)
        except Exception as e:
            cor = 0
        correct_answers.append(cor)
    return max(correct_answers)

def evaluate_performance_math(data, model_func, system_prompt, metric='best', num_samples=1):
    perf = 0
    total = len(data)
    
    if metric == 'best':
        metric = best_answer_accuracy
    elif metric == 'avg':
        metric = lambda model_answers, truth: np.mean([float(ans == truth) for ans in model_answers])
    else:
        assert isinstance(metric, function), f"Unknown string metric: {metric}"

    for idx, row in tqdm(data.iterrows(), total=total):
        question = row['Problem']
        ground_truth = row['Answer']
        prompt = system_prompt.format(question)
        
        model_outputs = model_func(prompt, num_samples=num_samples)
        model_answers = [extract_answer(output) for output in model_outputs]
        perf += metric(model_answers, ground_truth)
    return perf / total

################################
######### Display & Viz ########
################################

from IPython.display import Markdown
import textwrap

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))