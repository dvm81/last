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
from openai import AsyncAzureOpenAI
import hashlib
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import spacy
from spacy.tokens import Doc
from spacy.language import Language
import aiohttp
from aiohttp import ClientSession, TCPConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for storing article text by hash
_article_cache = {}

# Connection pooling configuration
MAX_CONNECTIONS = 10
CONNECTION_TIMEOUT = 30
KEEP_ALIVE_TIMEOUT = 60

# Chunking configuration
CHUNK_SIZE = 2000  # characters per chunk
CHUNK_OVERLAP = 200  # characters overlap between chunks
MAX_CONCURRENT_REQUESTS = 5  # Maximum concurrent API requests
BATCH_SIZE = 10  # Number of chunks to process in one batch

class AzureOpenAIClient:
    def __init__(self, api_key: str, api_version: str, azure_endpoint: str, deployment_name: str):
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.deployment_name = deployment_name
        self._session = None
        self._client = None
        
    async def __aenter__(self):
        # Create a connection pool
        connector = TCPConnector(
            limit=MAX_CONNECTIONS,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True
        )
        
        # Create aiohttp session with connection pooling
        self._session = ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(
                total=CONNECTION_TIMEOUT,
                connect=10,
                sock_read=10
            )
        )
        
        # Initialize Azure OpenAI client with the session
        self._client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            http_client=self._session
        )
        return self._client
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            
    @property
    def client(self):
        if not self._client:
            raise RuntimeError("Client not initialized. Use async with context.")
        return self._client

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    # Add custom entity ruler for company patterns
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [
        {"label": "ORG", "pattern": [{"LOWER": {"IN": ["inc", "corp", "ltd", "llc", "ltd.", "inc.", "corp."]}}]},
        {"label": "ORG", "pattern": [{"LOWER": {"IN": ["company", "corporation", "limited", "group", "holdings"]}}]},
        {"label": "ORG", "pattern": [{"LOWER": {"IN": ["plc", "pty", "pty.", "pty.ltd", "pty.ltd."]}}]},
    ]
    ruler.add_patterns(patterns)
except Exception as e:
    logger.error(f"Error loading spaCy model: {e}")
    nlp = None

def extract_companies_spacy(text: str) -> List[dict]:
    """Extract companies using spaCy's NER capabilities."""
    if nlp is None:
        logger.error("spaCy model not loaded")
        return []
        
    try:
        doc = nlp(text)
        companies = []
        
        # Extract organizations and companies
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                # Clean and normalize the company name
                company_name = ent.text.strip()
                if len(company_name) > 1:  # Avoid single character matches
                    companies.append({
                        "Word": company_name,
                        "RIC": {"value": "", "confidence": "none"},
                        "BBTicker": {"value": "", "confidence": "none"},
                        "Symbol": {"value": "", "confidence": "none"},
                        "ISIN": {"value": "", "confidence": "none"},
                        "SEDOL": {"value": "", "confidence": "none"},
                        "IssueName": {"value": company_name, "confidence": "medium"}
                    })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_companies = []
        for company in companies:
            company_id = company["Word"].lower()
            if company_id not in seen:
                seen.add(company_id)
                unique_companies.append(company)
        
        return unique_companies
        
    except Exception as e:
        logger.error(f"Error in spaCy extraction: {e}")
        return []

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

async def process_chunks_in_batch(chunks: List[str], llm, function_schema: dict) -> List[List[dict]]:
    """Process multiple chunks in a single batch request."""
    try:
        # Create a single system message
        system_message = {
            "role": "system",
            "content": EXTRACTION_SYSTEM_PROMPT
        }

        # Create user messages for each chunk
        user_messages = [
            {
                "role": "user",
                "content": EXTRACTION_HUMAN_PROMPT.format(article_text=chunk)
            }
            for chunk in chunks
        ]

        # Combine system and user messages
        messages = [system_message] + user_messages

        # Make a single API call with all messages
        response = await llm.chat.completions.create(
            model=AZURE_CONFIG["deployment_name"],
            messages=messages,
            functions=[function_schema],
            function_call={"name": function_schema["name"]},
            temperature=0.1,
        )

        # Parse the responses for each chunk
        results = []
        for choice in response.choices:
            function_args = json.loads(choice.message.function_call.arguments)
            results.append(function_args.get("companies", []))

        return results

    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        # Return empty results for each chunk on error
        return [[] for _ in chunks]

async def extract_companies_from_text_async_batch(article_text: str, llm, max_chunk_tokens: int = 1000, batch_size: int = 2) -> List[dict]:
    """Extract companies from text using optimized batch processing."""
    try:
        # Clean and prepare text
        article_text = article_text.strip()
        
        # Split text into chunks
        chunks = chunk_text(article_text)
        total_chunks = len(chunks)
        logger.info(f"Split article into {total_chunks} chunks")
        
        # Process chunks in batches
        all_results = []
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_results = await process_chunks_in_batch(batch, llm, EXTRACTION_FUNCTION)
            all_results.extend(batch_results)
            logger.info(f"Processed batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
        
        # Merge and deduplicate results
        companies = list(chain.from_iterable(all_results))
        unique_companies = []
        seen = set()
        
        for company in companies:
            company_id = company.get("Word", "").lower()
            if company_id and company_id not in seen:
                seen.add(company_id)
                unique_companies.append(company)
        
        logger.info(f"Found {len(unique_companies)} unique companies across {total_chunks} chunks")
        return unique_companies
        
    except Exception as e:
        logger.error(f"Error in company extraction: {e}")
        return []

# Hybrid approach that combines individual and batch processing
async def extract_companies_from_text_async_hybrid(article_text: str, llm, max_chunk_tokens: int = 1000, batch_size: int = 2) -> List[dict]:
    """Extract companies using a hybrid approach that adapts to chunk size."""
    try:
        # Clean and prepare text
        article_text = article_text.strip()
        
        # Split text into chunks
        chunks = chunk_text(article_text)
        total_chunks = len(chunks)
        logger.info(f"Split article into {total_chunks} chunks")
        
        # Process chunks based on size
        all_results = []
        
        # For small chunks, use batch processing
        if total_chunks <= batch_size * 2:
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i:i + batch_size]
                batch_results = await process_chunks_in_batch(batch, llm, EXTRACTION_FUNCTION)
                all_results.extend(batch_results)
                logger.info(f"Processed batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
        else:
            # For larger articles, process chunks individually for better reliability
            processor = ChunkProcessor(llm)
            for i in range(0, total_chunks, BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                batch_results = await processor.process_chunks_batch(
                    batch,
                    EXTRACTION_FUNCTION,
                    EXTRACTION_SYSTEM_PROMPT,
                    EXTRACTION_HUMAN_PROMPT
                )
                all_results.extend(batch_results)
                logger.info(f"Processed batch {i//BATCH_SIZE + 1}/{(total_chunks + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        # Merge and deduplicate results
        companies = list(chain.from_iterable(all_results))
        unique_companies = []
        seen = set()
        
        for company in companies:
            company_id = company.get("Word", "").lower()
            if company_id and company_id not in seen:
                seen.add(company_id)
                unique_companies.append(company)
        
        logger.info(f"Found {len(unique_companies)} unique companies across {total_chunks} chunks")
        return unique_companies
        
    except Exception as e:
        logger.error(f"Error in company extraction: {e}")
        return []

class ChunkProcessor:
    def __init__(self, llm_client: AzureOpenAIClient, max_concurrent: int = MAX_CONCURRENT_REQUESTS):
        self.llm_client = llm_client
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def process_chunk(
        self, 
        chunk: str, 
        function_schema: dict,
        system_prompt: str,
        human_prompt_template: str,
        **kwargs
    ) -> List[dict]:
        """Process a single chunk with rate limiting and connection pooling."""
        async with self.semaphore:
            try:
                # Format the human prompt with the chunk and any additional kwargs
                human_prompt = human_prompt_template.format(
                    article_text=chunk,
                    **kwargs
                )
                
                response = await self.llm_client.client.chat.completions.create(
                    model=AZURE_CONFIG["deployment_name"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": human_prompt}
                    ],
                    functions=[function_schema],
                    function_call={"name": function_schema["name"]},
                    temperature=0.1
                )
                
                function_args = json.loads(response.choices[0].message.function_call.arguments)
                return function_args.get("companies" if "companies" in function_args else "verified", [])
                
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                return []
                
    async def process_chunks_batch(
        self, 
        chunks: List[str], 
        function_schema: dict,
        system_prompt: str,
        human_prompt_template: str,
        **kwargs
    ) -> List[List[dict]]:
        """Process a batch of chunks concurrently."""
        tasks = [
            self.process_chunk(
                chunk, 
                function_schema,
                system_prompt,
                human_prompt_template,
                **kwargs
            ) 
            for chunk in chunks
        ]
        return await asyncio.gather(*tasks)

async def verify_companies_from_text_async(article: str, company_list: List[dict], llm) -> List[dict]:
    """Verify companies using optimized parallel processing."""
    try:
        # Split text into chunks
        chunks = chunk_text(article)
        total_chunks = len(chunks)
        logger.info(f"Split article into {total_chunks} chunks for verification")
        
        # Initialize chunk processor
        processor = ChunkProcessor(llm)
        
        # Process chunks in batches
        results = []
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            
            # Process batch concurrently
            batch_results = await processor.process_chunks_batch(
                batch,
                VERIFICATION_FUNCTION,
                VERIFICATION_SYSTEM_PROMPT,
                VERIFICATION_HUMAN_PROMPT,
                company_list=json.dumps(company_list, indent=2)
            )
            results.extend(batch_results)
            logger.info(f"Verified batch {i//BATCH_SIZE + 1}/{(total_chunks + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        # Merge and deduplicate verification results
        verified_companies = []
        seen_companies = set()
        
        for chunk_results in results:
            for company in chunk_results:
                company_id = company.get("MasterId")
                if company_id and company_id not in seen_companies:
                    seen_companies.add(company_id)
                    verified_companies.append(company)
        
        logger.info(f"Verified {len(verified_companies)} unique companies across {total_chunks} chunks")
        return verified_companies
        
    except Exception as e:
        logger.error(f"Error in company verification: {e}")
        return []

async def process_article_async(article_text: str, company_database: pd.DataFrame, llm_client: AzureOpenAIClient, use_cache: bool = True) -> tuple[List[dict], dict]:
    """Process article with caching, metrics tracking, and connection pooling."""
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
            companies = await cached_extract_companies(text_hash, llm_client)
        else:
            companies = await extract_companies_from_text_async_hybrid(article_text, llm_client)
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
            llm_client
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

# Example usage with connection pooling
async def process_articles_batch(articles: List[str], company_database: pd.DataFrame) -> List[tuple[List[dict], dict]]:
    """Process multiple articles using connection pooling."""
    async with AzureOpenAIClient(
        api_key=AZURE_CONFIG["api_key"],
        api_version=AZURE_CONFIG["api_version"],
        azure_endpoint=AZURE_CONFIG["azure_endpoint"],
        deployment_name=AZURE_CONFIG["deployment_name"]
    ) as llm_client:
        tasks = [
            process_article_async(article, company_database, llm_client)
            for article in articles
        ]
        return await asyncio.gather(*tasks)

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

@dataclass
class SearchConfig:
    limit_results_star: int = 5
    sim_threshold: float = 0.7
    limit_res_vs: int = 3
    limit_spec_results: int = 2
    cache_size: int = 1000
    batch_size: int = 100

class SearchIndex:
    """Efficient search index for company lookups."""
    def __init__(self, table: pd.DataFrame):
        self.table = table
        self.indices = {}
        self._build_indices()
        
    def _build_indices(self):
        """Build search indices once."""
        for field in ['RIC', 'BBTicker', 'Symbol', 'ISIN', 'SEDOL', 'IssueName']:
            if field in self.table.columns:
                # Create case-insensitive index
                self.indices[field] = self.table.set_index(
                    self.table[field].str.lower(),
                    drop=False  # Keep the original column
                )
                
    def get_matches(self, field: str, value: str, limit: int = 5) -> pd.DataFrame:
        """Get matches for a field value."""
        if field not in self.indices:
            return pd.DataFrame()
            
        try:
            value = value.lower()
            matches = self.indices[field].loc[value].head(limit)
            return matches
        except KeyError:
            return pd.DataFrame()
            
    def get_partial_matches(self, field: str, value: str, limit: int = 5) -> pd.DataFrame:
        """Get partial matches for a field value."""
        if field not in self.indices:
            return pd.DataFrame()
            
        try:
            base_value = value.strip().lower().split('.')[0]
            matches = self.indices[field].index.str.startswith(base_value)
            if matches.any():
                return self.indices[field][matches].head(limit)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

async def process_company_batch(
    companies: List[dict],
    search_index: SearchIndex,
    config: SearchConfig
) -> List[dict]:
    """Process a batch of companies efficiently."""
    results = []
    
    for company in companies:
        word_orig = company["Word"]
        word = word_orig.upper()
        
        # Extract identifiers
        identifiers = get_identifiers(company)
        subresult_list = []
        
        if len(identifiers) == 0:
            # Vector search on original word
            word_vector_results = retrieve_similar_entities(
                word_orig, 
                search_index.table, 
                limit=config.limit_res_vs, 
                threshold=config.sim_threshold
            )
            
            # Check IssueName if different from original word
            issueName_value = company.get("IssueName", {}).get("value")
            if issueName_value and issueName_value.lower() != word_orig.lower():
                word_vector_results.extend(
                    retrieve_similar_entities(
                        issueName_value, 
                        search_index.table, 
                        limit=config.limit_res_vs, 
                        threshold=config.sim_threshold
                    )
                )
            
            # Search in DB if vector results found
            if word_vector_results and any(x.get('IssueName') == word_orig for x in word_vector_results):
                matches = search_index.get_matches('IssueName', word_orig, config.limit_results_star)
                if not matches.empty:
                    matches['Word'] = word_orig
                    results.extend(matches.to_dict('records'))
            
            # Process vector search results
            for vector_result in word_vector_results:
                if isinstance(vector_result, dict) and 'IssueName' in vector_result:
                    issue_name = vector_result['IssueName']
                elif isinstance(vector_result, list) and len(vector_result) > 0 and isinstance(vector_result[0], dict):
                    issue_name = vector_result[0].get('IssueName')
                else:
                    continue
                
                if issue_name:
                    matches = search_index.get_matches('IssueName', issue_name, config.limit_results_star)
                    if not matches.empty:
                        matches['Word'] = word_orig
                        results.extend(matches.to_dict('records'))
        
        else:
            # Start with most confident identifier
            field_name, value = identifiers[0]
            matches = search_index.get_matches(field_name, value, config.limit_results_star)
            if not matches.empty:
                matches['Word'] = word_orig
                subresult_list = matches.to_dict('records')
            
            # Check for partial matches
            for field_name, value in identifiers:
                partial_matches = search_index.get_partial_matches(field_name, value, config.limit_results_star)
                if not partial_matches.empty:
                    partial_matches['Word'] = word_orig
                    subresult_list.extend(partial_matches.to_dict('records'))
        
        # Process speculative results
        limit_left = config.limit_spec_results
        speculative_results = []
        
        for result in subresult_list:
            if (word == result.get('RIC') or 
                word.lower() == result.get('BBTicker', '').lower() or 
                word.lower() == result.get('SEDOL', '').lower() or 
                word.lower() == result.get('ISIN', '').lower()):
                results.append(result)
                limit_left -= 1
            else:
                speculative_results.append(result)
        
        results.extend(speculative_results[:limit_left])
    
    return results

def hybrid_search(companies, config: SearchConfig = SearchConfig()):
    """
    Optimized hybrid search implementation that reduces database queries and improves performance.
    
    Args:
        companies: List of company dictionaries to search
        config: SearchConfig object with search parameters
        
    Returns:
        DataFrame with search results
    """
    # Pre-process all companies to extract identifiers once
    processed_companies = []
    for company in companies:
        # Get identifiers using existing function
        identifiers = get_identifiers(company)
        
        processed_companies.append({
            'word_orig': company['Word'],
            'word': company['Word'].upper(),
            'identifiers': identifiers,
            'issue_name': company.get("IssueName", {}).get("value", "")
        })
    
    # Batch process companies by identifier type
    results_list = []
    
    # 1. Process companies with identifiers first (most efficient)
    companies_with_ids = [c for c in processed_companies if c['identifiers']]
    if companies_with_ids:
        # Group by identifier type to reduce database queries
        for field in ["RIC", "BBTicker", "Symbol", "ISIN", "SEDOL"]:
            field_companies = [c for c in companies_with_ids if any(id[0] == field for id in c['identifiers'])]
            if not field_companies:
                continue
                
            # Get all values for this field
            values = [id[1] for c in field_companies for id in c['identifiers'] if id[0] == field]
            
            # Single database query for all values using existing function
            if values:
                for value in values:
                    results = query_duckdb_table_ins('instruments', field, value, top_n=config.limit_results_star)
                    if results is not None and not results.empty:
                        # Map results back to original companies
                        for _, result in results.iterrows():
                            for company in field_companies:
                                if any(id[1] == str(result[field]).upper() for id in company['identifiers'] if id[0] == field):
                                    result_dict = result.to_dict()
                                    result_dict['Word'] = company['word_orig']
                                    results_list.append(result_dict)
    
    # 2. Process companies without identifiers using vector search
    companies_without_ids = [c for c in processed_companies if not c['identifiers']]
    if companies_without_ids:
        # Batch vector search for all companies without identifiers
        all_vector_results = []
        for company in companies_without_ids:
            # Try issue name first if different from word
            if company['issue_name'] and company['issue_name'].casefold() != company['word_orig'].casefold():
                vector_results = retrieve_similar_entities(
                    company['issue_name'], 
                    table, 
                    limit=config.limit_res_vs, 
                    threshold=config.sim_threshold
                )
                all_vector_results.extend((company, result) for result in vector_results)
            
            # Fall back to word search
            vector_results = retrieve_similar_entities(
                company['word_orig'], 
                table, 
                limit=config.limit_res_vs, 
                threshold=config.sim_threshold
            )
            all_vector_results.extend((company, result) for result in vector_results)
        
        # Process vector results in batches
        if all_vector_results:
            # Group by company for efficient processing
            company_results = {}
            for company, result in all_vector_results:
                if company['word_orig'] not in company_results:
                    company_results[company['word_orig']] = []
                company_results[company['word_orig']].append(result)
            
            # Process each company's results
            for word_orig, results in company_results.items():
                limit_left = config.limit_spec_results
                for result in results:
                    if limit_left <= 0:
                        break
                        
                    # Check for direct matches using existing function
                    for field in ["RIC", "BBTicker", "Symbol", "ISIN", "SEDOL"]:
                        if field in result and word_orig.casefold() == str(result[field]).casefold():
                            result['Word'] = word_orig
                            results_list.append(result)
                            limit_left -= 1
                            break
    
    # Create final DataFrame and deduplicate efficiently
    if results_list:
        final_results = pd.DataFrame(results_list)
        
        # More efficient deduplication
        final_results = final_results.drop_duplicates(subset=['Word', 'ISIN'])
        
        # Sort by Word and AverageVolume
        final_results = final_results.sort_values(
            by=["Word", "AverageVolume"], 
            ascending=[False, False]
        )
    else:
        final_results = pd.DataFrame()
    
    return final_results 
