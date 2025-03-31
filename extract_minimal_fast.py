from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import json
import logging
from datetime import datetime
import time
import pandas as pd
from openai import OpenAI
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple models for structured output
class CompanyMention(BaseModel):
    Word: str
    IssueName: Optional[str] = None
    RIC: Optional[str] = None

class ExtractionResponse(BaseModel):
    companies: List[CompanyMention]

class VerifiedCompany(BaseModel):
    Word: str
    RIC: str
    MasterId: str

@dataclass
class SearchConfig:
    limit_results_star: int = 5
    sim_threshold: float = 0.7
    limit_res_vs: int = 3

# Minimal, focused prompts
EXTRACTION_PROMPT = """Extract ONLY company names, their RIC (if highly confident), and issue names.
Return in this exact format:
{
  "companies": [
    {
      "Word": "exact text found",
      "IssueName": "full company name if known",
      "RIC": "RIC code if highly confident"
    }
  ]
}
Only include RIC if very confident."""

VERIFICATION_PROMPT = """For these potential matches, return only verified matches in this exact format:
{
  "verified": [
    {
      "Word": "original text",
      "RIC": "verified RIC",
      "MasterId": "database ID"
    }
  ]
}
Include only high-confidence matches."""

def query_duckdb_table_case_ins(table_name: str, field: str, value: str, top_n: int = 5) -> pd.DataFrame:
    """Query DuckDB table with case-insensitive matching."""
    query = f"""
    SELECT *
    FROM {table_name}
    WHERE LOWER({field}) = LOWER('{value}')
    LIMIT {top_n}
    """
    return duckdb.query(query).df()

def retrieve_similar_entities(word: str, table: pd.DataFrame, limit: int = 3, threshold: float = 0.7) -> List[dict]:
    """Placeholder for vector similarity search."""
    # Implement your vector similarity search here
    return []

def hybrid_search(companies: List[CompanyMention], table: pd.DataFrame, config: SearchConfig = SearchConfig()) -> pd.DataFrame:
    """
    Simplified hybrid search focusing only on RIC and IssueName
    """
    results_list = []

    for company in companies:
        word_orig = company.Word
        ric = company.RIC
        issue_name = company.IssueName

        # First try exact match with RIC if available
        if ric:
            matches = query_duckdb_table_case_ins('instruments', 'RIC', ric, top_n=config.limit_results_star)
            if not matches.empty:
                matches['Word'] = word_orig
                results_list.extend(matches.to_dict('records'))
                continue

        # If no RIC match or no RIC, try IssueName
        if issue_name:
            # Try exact match first
            matches = query_duckdb_table_case_ins('instruments', 'IssueName', issue_name, top_n=config.limit_results_star)
            if not matches.empty:
                matches['Word'] = word_orig
                results_list.extend(matches.to_dict('records'))
                continue

            # If no exact match, try vector search on IssueName
            vector_results = retrieve_similar_entities(
                issue_name, 
                table, 
                limit=config.limit_res_vs, 
                threshold=config.sim_threshold
            )
            
            if vector_results:
                for result in vector_results:
                    if isinstance(result, dict) and 'IssueName' in result:
                        matches = query_duckdb_table_case_ins(
                            'instruments', 
                            'IssueName', 
                            result['IssueName'], 
                            top_n=config.limit_results_star
                        )
                        if not matches.empty:
                            matches['Word'] = word_orig
                            results_list.extend(matches.to_dict('records'))

    # Create final DataFrame and remove duplicates
    if results_list:
        final_results = pd.DataFrame(results_list)
        final_results.drop_duplicates(inplace=True)
    else:
        final_results = pd.DataFrame()

    return final_results

class FastExtractor:
    def __init__(self, llm: OpenAI):
        self.llm = llm
        
    def extract_companies(self, text: str) -> List[CompanyMention]:
        """Step 1: Initial extraction of companies"""
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            
            result = ExtractionResponse.model_validate_json(
                response.choices[0].message.content
            )
            return result.companies
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return []

    def verify_companies(self, potential_matches: List[dict]) -> List[VerifiedCompany]:
        """Step 3: Final verification of matches"""
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": VERIFICATION_PROMPT},
                    {"role": "user", "content": json.dumps(potential_matches, indent=2)}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return [VerifiedCompany(**company) for company in result["verified"]]
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return []

def process_article(text: str, database: pd.DataFrame, llm: OpenAI) -> List[dict]:
    """Fast, streamlined processing pipeline with hybrid search"""
    start_time = time.time()
    extractor = FastExtractor(llm)
    
    # Step 1: Extract companies
    companies = extractor.extract_companies(text)
    if not companies:
        return []
        
    # Step 2: Hybrid search for matches
    potential_matches = hybrid_search(companies, database)
    if potential_matches.empty:
        return []
        
    # Step 3: Verify matches
    verified_companies = extractor.verify_companies(potential_matches.to_dict('records'))
    
    end_time = time.time()
    logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    return [company.model_dump() for company in verified_companies]

# Usage example:
"""
llm = OpenAI()
database = pd.DataFrame(...)  # your company database
results = process_article("Your article text here", database, llm)
"""
