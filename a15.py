async def process_chunks_in_batch(chunks: List[str], llm, function_schema: dict) -> List[List[dict]]:
    """
    Process multiple chunks in a single batch request.
    """
    try:
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
            model=model_name,  # or deployment_id, depending on your Azure setup
            messages=messages,
            functions=[function_schema],
            function_call={"name": function_schema["name"]},
            temperature=0.0
        )

        # Parse the responses for each chunk
        results = []
        for choice in response.choices:
            try:
                # Check if function_call exists and has arguments
                if hasattr(choice.message, 'function_call') and choice.message.function_call and hasattr(choice.message.function_call, 'arguments'):
                    function_args = json.loads(choice.message.function_call.arguments)
                    # Ensure companies is a list
                    companies = function_args.get("companies", [])
                    if not isinstance(companies, list):
                        logger.warning(f"Expected companies to be a list, got {type(companies)}")
                        companies = []
                    results.append(companies)
                else:
                    logger.warning("No function_call or arguments found in response")
                    results.append([])
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON: {e}")
                results.append([])
            except Exception as e:
                logger.error(f"Error processing choice: {e}")
                results.append([])

        return results

    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return [[] for _ in chunks]  # Return empty results for each chunk on error


async def merge_chunk_results(chunk_results: List[List[dict]]) -> List[dict]:
    """
    Merge results from multiple chunks, removing duplicates.
    """
    try:
        # Flatten the list of results
        all_companies = []
        for companies in chunk_results:
            if isinstance(companies, list):
                all_companies.extend(companies)
            else:
                logger.warning(f"Expected list of companies, got {type(companies)}")
        
        # Remove duplicates based on company name
        unique_companies = []
        seen_names = set()
        
        for company in all_companies:
            if not isinstance(company, dict):
                logger.warning(f"Expected company to be a dict, got {type(company)}")
                continue
                
            company_name = company.get("Word", "")
            if company_name and company_name not in seen_names:
                seen_names.add(company_name)
                unique_companies.append(company)
        
        return unique_companies
    
    except Exception as e:
        logger.error(f"Error merging chunk results: {e}")
        return []


async def extract_companies_from_text_async_batch(article_text: str, llm, max_chunk_tokens: int = 1000, batch_size: int = 2) -> List[dict]:
    try:
        article_text = ensure_text_is_string(article_text)
        chunk_texts = chunk_text_by_tokens(text=article_text, max_chunk_tokens=max_chunk_tokens, model_encoding="cl100k_base")

        # Process chunks in batches
        batch_size = batch_size  # Adjust batch size based on API limits and efficiency
        tasks = []
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i: i + batch_size]
            tasks.append(process_chunks_in_batch(batch, llm, EXTRACTION_FUNCTION))

        batch_results = await asyncio.gather(*tasks)

        # Flatten the list of results
        chunk_results = [company for batch in batch_results for company in batch]
        companies = await merge_chunk_results(chunk_results)
        logger.info(f"Found {len(companies)} unique companies across all chunks")

        return companies

    except Exception as e:
        logger.error(f"Error in company extraction: {e}")
        return []
