import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline

class HuggingFaceLLM:
    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 512, repetition_penalty: float = 1.1):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self._load_model()

    def _load_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                quantization_config=bnb_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.text_generation_pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            return_full_text=True,
            max_new_tokens=self.max_tokens,
        )

        self.llm = HuggingFacePipeline(pipeline=self.text_generation_pipeline)

    def __call__(self, *args, **kwargs):
        return self.llm(*args, **kwargs)
  