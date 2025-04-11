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
from openai import AzureOpenAI
import hashlib
import asyncio
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for storing article text by hash
_article_cache = {}

# Chunking configuration
CHUNK_SIZE = 2000  # characters per chunk
CHUNK_OVERLAP = 200  # characters overlap between chunks

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        if end >= text_length:
            chunks.append(text[start:])
            break
            
        # Try to find a sentence boundary
        next_period = text.find('.', end - 100, end + 100)
        if next_period != -1:
            end = next_period + 1
            
        chunks.append(text[start:end])
        start = end - overlap
        
    return chunks

async def process_chunk_async(chunk: str, llm, function_schema: dict) -> List[dict]:
    """Process a single chunk asynchronously."""
    try:
        response = await asyncio.to_thread(
            llm.chat.completions.create,
            model=AZURE_CONFIG["deployment_name"],
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": EXTRACTION_HUMAN_PROMPT.format(article_text=chunk)}
            ],
            functions=[function_schema],
            function_call={"name": function_schema["name"]},
            temperature=0.1
        )
        
        # Extract function call arguments
        function_args = json.loads(response.choices[0].message.function_call.arguments)
        return function_args.get("companies", [])
        
    except Exception as e:
        logger.error(f"Error processing chunk: {e}")
        return []

async def merge_chunk_results(chunk_results: List[List[dict]]) -> List[dict]:
    """Merge results from multiple chunks, removing duplicates."""
    all_companies = []
    seen_words = set()
    
    for chunk in chunk_results:
        for company in chunk:
            word = company["Word"]
            if word not in seen_words:
                seen_words.add(word)
                all_companies.append(company)
                
    return all_companies

# Azure OpenAI Configuration
AZURE_CONFIG = {
    "api_key": "YOUR_AZURE_API_KEY",
    "api_version": "2024-02-15-preview",
    "azure_endpoint": "YOUR_AZURE_ENDPOINT",
    "deployment_name": "YOUR_DEPLOYMENT_NAME"
}

# Initialize Azure OpenAI client
def get_azure_client():
    return AzureOpenAI(
        api_key=AZURE_CONFIG["api_key"],
        api_version=AZURE_CONFIG["api_version"],
        azure_endpoint=AZURE_CONFIG["azure_endpoint"]
    )

class ConfidenceLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

# Define function schemas for GPT-4
EXTRACTION_FUNCTION = {
    "name": "extract_companies",
    "description": "Extract company mentions from text with their identifiers and confidence levels",
    "parameters": {
        "type": "object",
        "properties": {
            "companies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "Word": {"type": "string", "description": "The exact text found in the article"},
                        "RIC": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                                "confidence": {"type": "string", "enum": ["none", "low", "medium", "high", "very_high"]}
                            }
                        },
                        "BBTicker": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                                "confidence": {"type": "string", "enum": ["none", "low", "medium", "high", "very_high"]}
                            }
                        },
                        "Symbol": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                                "confidence": {"type": "string", "enum": ["none", "low", "medium", "high", "very_high"]}
                            }
                        },
                        "ISIN": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                                "confidence": {"type": "string", "enum": ["none", "low", "medium", "high", "very_high"]}
                            }
                        },
                        "SEDOL": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                                "confidence": {"type": "string", "enum": ["none", "low", "medium", "high", "very_high"]}
                            }
                        },
                        "IssueName": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                                "confidence": {"type": "string", "enum": ["none", "low", "medium", "high", "very_high"]}
                            }
                        }
                    }
                }
            }
        }
    }
}

VERIFICATION_FUNCTION = {
    "name": "verify_companies",
    "description": "Verify and match companies from the article against the company database",
    "parameters": {
        "type": "object",
        "properties": {
            "verified": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "MasterId": {"type": "string"},
                        "CompanyName": {"type": "string"},
                        "CountryOfListing": {"type": "string"},
                        "RIC": {"type": "string"},
                        "BBTicker": {"type": "string"},
                        "Symbol": {"type": "string"},
                        "ISIN": {"type": "string"},
                        "SEDOL": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["none", "low", "medium", "high", "very_high"]}
                    }
                }
            }
        }
    }
}

# Optimized prompt templates
EXTRACTION_SYSTEM_PROMPT = """You are an expert entity extraction assistant. Extract company mentions from text.
For each company mention:
1. Extract the exact text found (Word)
2. If you can determine any identifiers (RIC, BBTicker, Symbol, ISIN, SEDOL, IssueName), include them with confidence levels
3. Prioritize recall over precision
4. Only include identifiers when you are confident about them"""

EXTRACTION_HUMAN_PROMPT = """Extract company mentions from this text:
{article_text}"""

VERIFICATION_SYSTEM_PROMPT = """You are a highly accurate company name disambiguation assistant.
Your task is to analyze an article and determine which companies from a provided list are being discussed.
For each company:
1. Verify if it matches a company in the database
2. Provide a confidence level for the match
3. Include all available identifiers
4. Pay special attention to the CountryOfListing for accuracy"""

VERIFICATION_HUMAN_PROMPT = """### Company List:
{company_list}

**Article:**
{article}

Match each company found in the article to the best match in the company database list.
Pay special attention to the respective CountryOfListing to be precise."""

class ExtractionMetrics:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.execution_time = None
        self.token_count = 0
        self.companies_found = 0
        self.extraction_time = 0
        self.search_time = 0
        self.verification_time = 0
        self.step_timings = {
            "extraction": {"start": None, "end": None, "duration": 0},
            "search": {"start": None, "end": None, "duration": 0},
            "verification": {"start": None, "end": None, "duration": 0}
        }
        
    def start(self):
        self.start_time = time.time()
        
    def end(self):
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        
    def start_step(self, step_name: str):
        """Start timing a specific step."""
        if step_name in self.step_timings:
            self.step_timings[step_name]["start"] = time.time()
            
    def end_step(self, step_name: str):
        """End timing a specific step and calculate duration."""
        if step_name in self.step_timings:
            self.step_timings[step_name]["end"] = time.time()
            self.step_timings[step_name]["duration"] = (
                self.step_timings[step_name]["end"] - 
                self.step_timings[step_name]["start"]
            )
        
    def get_metrics(self) -> dict:
        """Get all metrics as a dictionary."""
        return {
            "execution_time": self.execution_time,
            "token_count": self.token_count,
            "companies_found": self.companies_found,
            "step_timings": {
                step: {
                    "duration": timing["duration"],
                    "start": datetime.fromtimestamp(timing["start"]).strftime('%Y-%m-%d %H:%M:%S') if timing["start"] else None,
                    "end": datetime.fromtimestamp(timing["end"]).strftime('%Y-%m-%d %H:%M:%S') if timing["end"] else None
                }
                for step, timing in self.step_timings.items()
            },
            "start_time": datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S') if self.start_time else None,
            "end_time": datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S') if self.end_time else None
        }
        
    def log_metrics(self):
        """Log metrics to the logger."""
        metrics = self.get_metrics()
        logger.info(f"""
Extraction Metrics:
Total Execution Time: {metrics['execution_time']:.2f} seconds
Companies Found: {metrics['companies_found']}
Token Count: {metrics['token_count']}

Step Timings:
Extraction: {metrics['step_timings']['extraction']['duration']:.2f}s
Search: {metrics['step_timings']['search']['duration']:.2f}s
Verification: {metrics['step_timings']['verification']['duration']:.2f}s

Started: {metrics['start_time']}
Finished: {metrics['end_time']}
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

# Add a cache for storing article text by hash
_article_cache = {}

@lru_cache(maxsize=1000)
def cached_extract_companies(text_hash: str, llm) -> List[dict]:
    """Cached version of company extraction."""
    if text_hash not in _article_cache:
        logger.error(f"Cache miss for hash {text_hash}")
        return []
    return asyncio.run(extract_companies_from_text_async(_article_cache[text_hash], llm))

def parse_json_response(response: str) -> dict:
    """Parse JSON response from LLM, handling potential markdown formatting."""
    try:
        # Remove markdown code block markers if present
        cleaned_response = response.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Raw response: {response}")
        return {"companies": []}

async def extract_companies_from_text_async(article_text: str, llm) -> List[dict]:
    """Extract companies from text using Azure OpenAI with function calling and chunking."""
    try:
        # Split text into chunks
        chunks = chunk_text(article_text)
        logger.info(f"Split article into {len(chunks)} chunks")
        
        # Process chunks asynchronously
        tasks = [process_chunk_async(chunk, llm, EXTRACTION_FUNCTION) for chunk in chunks]
        chunk_results = await asyncio.gather(*tasks)
        
        # Merge results from all chunks
        companies = await merge_chunk_results(chunk_results)
        logger.info(f"Found {len(companies)} unique companies across all chunks")
        
        return companies
        
    except Exception as e:
        logger.error(f"Error in company extraction: {e}")
        return []

async def verify_companies_from_text_async(article: str, company_list: List[dict], llm) -> List[dict]:
    """Verify companies using Azure OpenAI with function calling and chunking."""
    try:
        # Split text into chunks
        chunks = chunk_text(article)
        logger.info(f"Split article into {len(chunks)} chunks for verification")
        
        # Process chunks asynchronously
        tasks = []
        for chunk in chunks:
            task = asyncio.to_thread(
                llm.chat.completions.create,
                model=AZURE_CONFIG["deployment_name"],
                messages=[
                    {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": VERIFICATION_HUMAN_PROMPT.format(
                        company_list=json.dumps(company_list, indent=2),
                        article=chunk
                    )}
                ],
                functions=[VERIFICATION_FUNCTION],
                function_call={"name": "verify_companies"},
                temperature=0.1
            )
            tasks.append(task)
            
        responses = await asyncio.gather(*tasks)
        
        # Merge verification results
        all_verified = []
        seen_companies = set()
        
        for response in responses:
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            verified = function_args.get("verified", [])
            
            for company in verified:
                company_id = company.get("MasterId")
                if company_id and company_id not in seen_companies:
                    seen_companies.add(company_id)
                    all_verified.append(company)
                    
        logger.info(f"Verified {len(all_verified)} unique companies across all chunks")
        return all_verified
        
    except Exception as e:
        logger.error(f"Error in company verification: {e}")
        return []

async def process_article_async(article_text: str, company_database: pd.DataFrame, llm, use_cache: bool = True) -> tuple[List[dict], dict]:
    """Process article with caching, metrics tracking, and async processing."""
    metrics = ExtractionMetrics()
    metrics.start()
    
    try:
        # Generate hash for caching
        text_hash = hashlib.md5(article_text.encode()).hexdigest()
        
        # Store article text in cache if caching is enabled
        if use_cache:
            _article_cache[text_hash] = article_text
        
        # Extract companies (with caching if enabled)
        metrics.start_step("extraction")
        if use_cache:
            companies = cached_extract_companies(text_hash, llm)
        else:
            companies = await extract_companies_from_text_async(article_text, llm)
        metrics.end_step("extraction")
            
        if not companies:
            return [], metrics.get_metrics()
            
        # Perform hybrid search
        metrics.start_step("search")
        potential_matches = hybrid_search(companies, company_database)
        metrics.end_step("search")
        
        if potential_matches.empty:
            return [], metrics.get_metrics()
            
        # Verify matches
        metrics.start_step("verification")
        verified_companies = await verify_companies_from_text_async(
            article_text,
            potential_matches.to_dict('records'),
            llm
        )
        metrics.end_step("verification")
        
        metrics.companies_found = len(verified_companies)
        return verified_companies, metrics.get_metrics()
        
    except Exception as e:
        logger.error(f"Error in article processing: {e}")
        return [], metrics.get_metrics()
        
    finally:
        metrics.end()
        metrics.log_metrics()

# Update the synchronous version to use the async version
def process_article(article_text: str, company_database: pd.DataFrame, llm, use_cache: bool = True) -> tuple[List[dict], dict]:
    """Synchronous wrapper for process_article_async."""
    return asyncio.run(process_article_async(article_text, company_database, llm, use_cache))

# Usage example:
"""
# Initialize Azure OpenAI client
llm = get_azure_client()

# Load your company database
company_database = pd.DataFrame(...)  # your company database

# Process an article
results, metadata = process_article(
    "Your article text here",
    company_database,
    llm,
    use_cache=True
)
""" 

llm = get_azure_client()
results, metadata = process_article(
    "Your article text here",
    company_database,
    llm,
    use_cache=True
) 
