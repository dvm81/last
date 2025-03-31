# extract_optimized_structured.py
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, TypeVar, Type
from pydantic import BaseModel, Field
import json
from functools import lru_cache
import logging
from datetime import datetime
import time
import pandas as pd
import duckdb
from dataclasses import dataclass
from openai import OpenAI

# Define Pydantic models for structured output
class ConfidenceLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class Identifier(BaseModel):
    value: Optional[str] = None
    confidence: ConfidenceLevel

class CompanyMention(BaseModel):
    Word: str
    # Only include the identifiers that were found with confidence
    identifiers: List[Dict[str, Identifier]] = Field(
        description="Include only the 2 identifiers with highest confidence (high or very_high). Skip any with none confidence.",
        max_items=2
    )

class ExtractionResponse(BaseModel):
    companies: List[CompanyMention] = Field(
        description="List of company mentions with their identifiers"
    )

class VerifiedCompany(BaseModel):
    MasterId: str
    CompanyName: str
    CountryOfListing: str
    RIC: Optional[str] = None
    BBTicker: Optional[str] = None
    Symbol: Optional[str] = None
    ISIN: Optional[str] = None
    SEDOL: Optional[str] = None
    confidence: ConfidenceLevel

T = TypeVar('T', bound=BaseModel)

def create_structured_prompt(model: Type[T]) -> str:
    """Create a structured prompt based on Pydantic model"""
    schema = model.model_json_schema()
    return f"""You must respond with valid JSON that matches this schema:
{json.dumps(schema, indent=2)}

The response must be parseable JSON that matches the schema exactly."""

def with_structured_output(llm: OpenAI, model: Type[T]):
    """Create a wrapper for structured output"""
    def structured_completion(messages: List[dict]) -> T:
        # Add schema instructions to system message
        schema_instruction = create_structured_prompt(model)
        if messages[0]["role"] == "system":
            messages[0]["content"] = f"{messages[0]['content']}\n\n{schema_instruction}"
        else:
            messages.insert(0, {"role": "system", "content": schema_instruction})
        
        response = llm.chat.completions.create(
            model="gpt-4",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        try:
            result = model.model_validate_json(response.choices[0].message.content)
            return result
        except Exception as e:
            logging.error(f"Failed to parse response: {e}")
            raise
    
    return structured_completion

class ExtractionMetrics:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.execution_time = None
        self.token_count = 0
        self.companies_found = 0
        
    def start(self):
        self.start_time = time.time()
        
    def end(self):
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        
    def log_metrics(self):
        logging.info(f"""
Extraction Metrics:
Execution Time: {self.execution_time:.2f} seconds
Companies Found: {self.companies_found}
Started: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}
Finished: {datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')}
        """)

# Optimized prompt templates
EXTRACTION_SYSTEM_PROMPT = """You are an expert entity extraction assistant. Extract company mentions from text.
For each company mention, return ONLY:
1. The exact Word found in the text
2. The single most confident identifier as 'top_identifier', including:
   - type: The type of identifier (RIC, BBTicker, SEDOL, ISIN, or Symbol)
   - value: The identifier value
   - confidence: ONLY "high" or "very_high"
3. IssueName if available

The response MUST be a valid JSON object with this exact structure:
{{
  "companies": [
    {{
      "Word": "Example Corp",
      "top_identifier": {{
        "type": "RIC",
        "value": "EXMP.O",
        "confidence": "very_high"
      }},
      "IssueName": {{
        "type": "IssueName",
        "value": "Example Corporation",
        "confidence": "high"
      }}
    }}
  ]
}}"""

EXTRACTION_HUMAN_PROMPT = """Extract company mentions from this text:
{article_text}"""

VERIFICATION_SYSTEM_PROMPT = """You are a highly accurate company name disambiguation assistant.
Your task is to analyze an article and determine which companies from a provided list are being discussed.
For each company, provide a confidence level (none, low, medium, high, very_high) indicating the likelihood that the company is addressed in the article."""

VERIFICATION_HUMAN_PROMPT = """### Company List:
{company_list}

**Article:**
{article}

Match each company found in the article to the best match in the company database list.
Pay special attention to the respective CountryOfListing to be precise."""

@dataclass
class SearchConfig:
    limit_results_star: int = 5
    sim_threshold: float = 0.7
    limit_res_vs: int = 3
    limit_spec_results: int = 2

def extract_companies_from_text(article_text: str, llm: OpenAI) -> List[dict]:
    metrics = ExtractionMetrics()
    metrics.start()
    
    try:
        messages = [
            {
                "role": "system", 
                "content": f"""You are an expert entity extraction assistant. Extract company mentions from text.
For each company mention, return ONLY:
1. The exact Word found in the text
2. The single most confident identifier as 'top_identifier', including:
   - type: The type of identifier (RIC, BBTicker, SEDOL, ISIN, or Symbol)
   - value: The identifier value
   - confidence: ONLY "high" or "very_high"
3. IssueName if available

The response MUST be a valid JSON object with this exact structure:
{{
  "companies": [
    {{
      "Word": "Example Corp",
      "top_identifier": {{
        "type": "RIC",
        "value": "EXMP.O",
        "confidence": "very_high"
      }},
      "IssueName": {{
        "type": "IssueName",
        "value": "Example Corporation",
        "confidence": "high"
      }}
    }}
  ]
}}"""
            },
            {
                "role": "user", 
                "content": f"Extract company mentions from this text:\n{article_text}"
            }
        ]
        
        response = llm.chat.completions.create(
            model="gpt-4",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        try:
            content = response.choices[0].message.content
            logging.debug(f"Raw LLM response: {content}")
            
            parsed_response = ExtractionResponse.model_validate_json(content)
            companies = [company.model_dump() for company in parsed_response.companies]
            metrics.companies_found = len(companies)
            
        except Exception as parse_error:
            logging.error(f"Failed to parse LLM response: {parse_error}")
            logging.error(f"Response content: {content}")
            companies = []
            
    except Exception as e:
        logging.error(f"Error during extraction: {str(e)}")
        companies = []
        
    finally:
        metrics.end()
        metrics.log_metrics()
        
    return companies

def verify_companies_from_text(article: str, company_list: List[dict], llm: OpenAI) -> List[dict]:
    try:
        messages = [
            {
                "role": "system", 
                "content": f"{VERIFICATION_SYSTEM_PROMPT}\n\nRespond with a JSON array matching this schema:\n{json.dumps(VerifiedCompany.model_json_schema(), indent=2)}"
            },
            {
                "role": "user", 
                "content": VERIFICATION_HUMAN_PROMPT.format(
                    article=article,
                    company_list=json.dumps(company_list, indent=2)
                )
            }
        ]
        
        response = llm.chat.completions.create(
            model="gpt-4",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        verified_companies = List[VerifiedCompany].model_validate_json(
            response.choices[0].message.content
        )
        return [company.model_dump() for company in verified_companies]
            
    except Exception as e:
        logging.error(f"Error during verification: {str(e)}")
        return []

def process_article(article_text: str, company_database: pd.DataFrame, llm: OpenAI, use_cache: bool = True) -> tuple[List[dict], dict]:
    """
    Process an article through the complete pipeline:
    1. Extract companies
    2. Hybrid search
    3. Verify matches
    """
    start_time = time.time()
    
    # Step 1: Extract companies
    if use_cache:
        text_hash = hash(article_text)
        extracted_companies = cached_extract_companies(text_hash, llm)
    else:
        extracted_companies = extract_companies_from_text(article_text, llm)
    
    # Step 2: Hybrid search
    matched_companies = hybrid_search(extracted_companies, company_database)
    
    # Step 3: Verify matches
    verified_companies = verify_companies_from_text(article_text, matched_companies.to_dict('records'), llm)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    metrics = {
        'execution_time': execution_time,
        'article_length': len(article_text),
        'companies_found': len(verified_companies),
        'timestamp': datetime.now(),
        'caching_used': use_cache
    }
    
    return verified_companies, metrics

@lru_cache(maxsize=1000)
def cached_extract_companies(text_hash: str, llm: OpenAI) -> List[dict]:
    return extract_companies_from_text(text_hash, llm)
