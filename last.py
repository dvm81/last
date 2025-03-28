from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import json
from functools import lru_cache
import logging
from datetime import datetime
import time
import pandas as pd
import duckdb
from dataclasses import dataclass

class ConfidenceLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

# Define the output structure for the initial extraction
EXTRACTION_SCHEMA = {
    "companies": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "Word": {"type": "string"},
                "RIC": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "optional": True},
                        "confidence": {"type": "string", "enum": ["none", "low", "medium", "high", "very_high"]}
                    }
                },
                "BBTicker": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "optional": True},
                        "confidence": {"type": "string", "enum": ["none", "low", "medium", "high", "very_high"]}
                    }
                },
                "Symbol": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "optional": True},
                        "confidence": {"type": "string", "enum": ["none", "low", "medium", "high", "very_high"]}
                    }
                },
                "ISIN": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "optional": True},
                        "confidence": {"type": "string", "enum": ["none", "low", "medium", "high", "very_high"]}
                    }
                },
                "SEDOL": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "optional": True},
                        "confidence": {"type": "string", "enum": ["none", "low", "medium", "high", "very_high"]}
                    }
                },
                "IssueName": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "optional": True},
                        "confidence": {"type": "string", "enum": ["none", "low", "medium", "high", "very_high"]}
                    }
                }
            }
        }
    }
}

# Define the output structure for verification
VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "MasterId": {"type": "string"},
        "CompanyName": {"type": "string"},
        "CountryOfListing": {"type": "string"},
        "RIC": {"type": "string", "optional": True},
        "BBTicker": {"type": "string", "optional": True},
        "Symbol": {"type": "string", "optional": True},
        "ISIN": {"type": "string", "optional": True},
        "SEDOL": {"type": "string", "optional": True},
        "confidence": {"type": "string", "enum": ["none", "low", "medium", "high", "very_high"]}
    }
}

# Optimized prompt templates
EXTRACTION_SYSTEM_PROMPT = """You are an expert entity extraction assistant. Extract company mentions from text.
Return a JSON array with company mentions, each containing "Word" and up to 2 identifiers with highest confidence.
Each identifier has "value" and "confidence" (one of: "none", "low", "medium", "high", "very_high").
Prioritize recall over precision.

The response must be a valid JSON object with a "companies" array containing company mentions.
Each company mention must include:
- Word: The exact text found
- Up to 2 identifiers (RIC, BBTicker, Symbol, ISIN, SEDOL, IssueName)
- Each identifier needs value and confidence level
- Set unused identifiers to null with "none" confidence"""

EXTRACTION_HUMAN_PROMPT = """Extract company mentions from this text:
{article_text}"""

VERIFICATION_SYSTEM_PROMPT = """You are a highly accurate company name disambiguation assistant.
Your task is to analyze an article and determine which companies from a provided list are being discussed.
For each company, provide a confidence level (none, low, medium, high, very_high) indicating the likelihood that the company is addressed in the article.

The response must be a valid JSON object containing:
- MasterId: The unique identifier
- CompanyName: The official company name
- CountryOfListing: The listing country
- Available identifiers (RIC, BBTicker, Symbol, ISIN, SEDOL)
- confidence: The confidence level for the match"""

VERIFICATION_HUMAN_PROMPT = """### Company List:
{company_list}

**Article:**
{article}

Match each company found in the article to the best match in the company database list.
Pay special attention to the respective CountryOfListing to be precise.
Provide your answers in JSON format, ensuring that each company includes all available fields along with a confidence level.
The field MasterId is an identifier that should be preserved in your answer.

Guidelines for confidence:
- none: Almost certain company not addressed
- low: Weak guess that the company might be addressed
- medium: Moderate confidence
- high: Fairly sure
- very_high: Completely sure"""

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

@dataclass
class SearchConfig:
    limit_results_star: int = 5
    sim_threshold: float = 0.7
    limit_res_vs: int = 3
    limit_spec_results: int = 2

def get_identifiers(company: dict) -> List[Tuple[str, str]]:
    """Extract identifiers with their confidence levels."""
    identifiers = []
    
    # Helper function to check confidence and add identifier
    def add_identifier(field_name: str, value: str, confidence: str):
        if value and confidence in ["high", "very_high"]:
            # Store confidence level as second element for sorting
            identifiers.append((field_name, confidence))
    
    # Check each identifier
    add_identifier("RIC", company.get("RIC", {}).get("value"), company.get("RIC", {}).get("confidence"))
    add_identifier("BBTicker", company.get("BBTicker", {}).get("value"), company.get("BBTicker", {}).get("confidence"))
    add_identifier("Symbol", company.get("Symbol", {}).get("value"), company.get("Symbol", {}).get("confidence"))
    add_identifier("ISIN", company.get("ISIN", {}).get("value"), company.get("ISIN", {}).get("confidence"))
    add_identifier("SEDOL", company.get("SEDOL", {}).get("value"), company.get("SEDOL", {}).get("confidence"))
    
    # Sort by confidence level (very_high > high)
    confidence_order = {"very_high": 2, "high": 1}
    return sorted(identifiers, key=lambda x: confidence_order.get(x[1], 0), reverse=True)

def retrieve_similar_entities(word: str, table: pd.DataFrame, limit: int = 3, threshold: float = 0.7) -> List[dict]:
    """Placeholder for vector similarity search."""
    # Implement your vector similarity search here
    # This should return a list of dictionaries with similar entities
    return []

def query_duckdb_table_case_ins(table_name: str, field: str, value: str, top_n: int = 5) -> pd.DataFrame:
    """Query DuckDB table with case-insensitive matching."""
    query = f"""
    SELECT *
    FROM {table_name}
    WHERE LOWER({field}) = LOWER('{value}')
    LIMIT {top_n}
    """
    return duckdb.query(query).df()

def query_duckdb_table_like_ins(table_name: str, field: str, value: str, top_n: int = 5) -> pd.DataFrame:
    """Query DuckDB table with case-insensitive LIKE matching."""
    query = f"""
    SELECT *
    FROM {table_name}
    WHERE LOWER({field}) LIKE LOWER('{value}%')
    LIMIT {top_n}
    """
    return duckdb.query(query).df()

def hybrid_search(companies: List[dict], table: pd.DataFrame, config: SearchConfig = SearchConfig()) -> pd.DataFrame:
    """
    Perform hybrid search between extracted companies and company database.
    """
    results_list = []

    for company in companies:
        word_orig = company["Word"]
        word = word_orig.upper()

        # Extract identifiers
        identifiers = get_identifiers(company)
        subresult_list = []

        if len(identifiers) == 0:
            # Vector search on original word
            word_vector_results = retrieve_similar_entities(word_orig, table, limit=config.limit_res_vs, threshold=config.sim_threshold)

            # Check IssueName if different from original word
            issueName_value = company.get("IssueName", {}).get("value")
            if issueName_value and issueName_value.lower() != word_orig.lower():
                word_vector_results.extend(retrieve_similar_entities(issueName_value, table, limit=config.limit_res_vs, threshold=config.sim_threshold))

            # Search in DB if vector results found
            if word_vector_results and any(x.get('IssueName') == word_orig for x in word_vector_results):
                star_res = query_duckdb_table_case_ins('instruments', 'IssueName', word_orig, top_n=config.limit_results_star).to_dict(orient='records')
                for record in star_res:
                    record['Word'] = word_orig
                results_list.extend(star_res)

            # Process vector search results
            for vector_result in word_vector_results:
                if isinstance(vector_result, dict) and 'IssueName' in vector_result:
                    issue_name = vector_result['IssueName']
                elif isinstance(vector_result, list) and len(vector_result) > 0 and isinstance(vector_result[0], dict):
                    issue_name = vector_result[0].get('IssueName')
                else:
                    continue

                if issue_name:
                    star_res = query_duckdb_table_case_ins('instruments', 'IssueName', issue_name, top_n=config.limit_results_star).to_dict(orient='records')
                    for record in star_res:
                        record['Word'] = word_orig
                    results_list.extend(star_res)

        else:
            # Start with most confident identifier
            field_name, value = identifiers[0]
            topid_star_result = query_duckdb_table_case_ins('instruments', field_name, value, top_n=config.limit_results_star).to_dict(orient='records')
            if topid_star_result:
                for record in topid_star_result:
                    record['Word'] = word_orig
                subresult_list = topid_star_result

            # Check for partial matches
            for field_name, value in identifiers:
                if word.strip().lower().split('.')[0] == value.strip().lower().split('.')[0]:
                    records = query_duckdb_table_like_ins('instruments', field_name, value, top_n=config.limit_results_star).to_dict(orient='records')
                    for record in records:
                        record['Word'] = word_orig
                        subresult_list.append(record)

        # Vector search if no results found
        if not subresult_list and not results_list:
            word_vector_result = retrieve_similar_entities(word_orig, table, limit=config.limit_res_vs, threshold=config.sim_threshold)
            if word_vector_result:
                if isinstance(word_vector_result, dict) and 'IssueName' in word_vector_result:
                    issue_name = word_vector_result['IssueName']
                elif isinstance(word_vector_result, list) and len(word_vector_result) > 0 and isinstance(word_vector_result[0], dict):
                    issue_name = word_vector_result[0].get('IssueName')

                if issue_name:
                    star_res = query_duckdb_table_case_ins('instruments', 'IssueName', issue_name, top_n=config.limit_results_star).to_dict(orient='records')
                    for record in star_res:
                        record['Word'] = word_orig
                    results_list.extend(star_res)

        # Process speculative results
        limit_left = config.limit_spec_results
        speculative_results = []

        for result in subresult_list:
            if (word == result.get('RIC') or 
                word.lower() == result.get('BBTicker', '').lower() or 
                word.lower() == result.get('SEDOL', '').lower() or 
                word.lower() == result.get('ISIN', '').lower()):
                results_list.append(result)
                limit_left -= 1
            else:
                speculative_results.append(result)

        results_list.extend(speculative_results[:limit_left])

    # Create final DataFrame and remove duplicates
    if results_list:
        final_results = pd.DataFrame(results_list)
        final_results.drop_duplicates(inplace=True)
    else:
        final_results = pd.DataFrame()

    return final_results

@lru_cache(maxsize=1000)
def cached_extract_companies(text_hash: str, llm) -> List[dict]:
    return extract_companies_from_text(text_hash, llm)

def parse_json_response(response: str) -> dict:
    """Parse JSON response and handle markdown code blocks."""
    # Remove markdown code block markers if present
    cleaned = response.replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON response: {str(e)}")
        logging.debug(f"Raw response: {response}")
        logging.debug(f"Cleaned response: {cleaned}")
        return {}

def extract_companies_from_text(article_text: str, llm) -> List[dict]:
    metrics = ExtractionMetrics()
    metrics.start()
    
    try:
        # Format messages directly
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": EXTRACTION_HUMAN_PROMPT.format(article_text=article_text)}
        ]
        
        # Get LLM response
        response = llm(messages)
        raw_output = response.choices[0].message.content if hasattr(response, 'choices') else str(response)
        
        # Parse response
        parsed_output = parse_json_response(raw_output)
        companies = parsed_output.get('companies', [])
        metrics.companies_found = len(companies)
            
    except Exception as e:
        logging.error(f"Error during extraction: {str(e)}")
        companies = []
        
    finally:
        metrics.end()
        metrics.log_metrics()
        
    return companies

def verify_companies_from_text(article: str, company_list: List[dict], llm) -> List[dict]:
    """Verify and match companies from the article against a company database."""
    try:
        # Format messages directly
        messages = [
            {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": VERIFICATION_HUMAN_PROMPT.format(
                article=article,
                company_list=json.dumps(company_list, indent=2)
            )}
        ]
        
        # Get LLM response
        response = llm(messages)
        raw_output = response.choices[0].message.content if hasattr(response, 'choices') else str(response)
        
        # Parse response
        verified_companies = parse_json_response(raw_output)
        return verified_companies
            
    except Exception as e:
        logging.error(f"Error during verification: {str(e)}")
        return []

def process_article(article_text: str, company_database: pd.DataFrame, llm, use_cache: bool = True) -> tuple[List[dict], dict]:
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
