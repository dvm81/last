async def process_chunks_in_batch(chunks: List[str], llm, function_schema: dict) -> List[List[dict]]:
    """
    Process multiple chunks in parallel using separate API calls for each chunk.
    This ensures each chunk gets its own dedicated response.
    """
    try:
        # Create separate tasks for each chunk
        tasks = []
        for chunk in chunks:
            # Create a task for each chunk with its own system and user message
            task = process_single_chunk(chunk, llm, function_schema)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in individual tasks
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing chunk {i}: {result}")
                processed_results.append([])
            else:
                processed_results.append(result)
        
        return processed_results

    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return [[] for _ in chunks]


async def process_single_chunk(chunk: str, llm, function_schema: dict) -> List[dict]:
    """
    Process a single chunk with its own dedicated API call.
    """
    try:
        # Create messages for this chunk only
        messages = [
            {
                "role": "system",
                "content": EXTRACTION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": EXTRACTION_HUMAN_PROMPT.format(article_text=chunk)
            }
        ]

        # Make a dedicated API call for this chunk
        response = await llm.chat.completions.create(
            model=model_name,  # or deployment_id, depending on your Azure setup
            messages=messages,
            functions=[function_schema],
            function_call={"name": function_schema["name"]},
            temperature=0.0
        )

        # Parse the response
        if hasattr(response.choices[0].message, 'function_call') and response.choices[0].message.function_call:
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            companies = function_args.get("companies", [])
            if not isinstance(companies, list):
                logger.warning(f"Expected companies to be a list, got {type(companies)}")
                return []
            return companies
        else:
            logger.warning("No function_call found in response")
            return []

    except Exception as e:
        logger.error(f"Error processing chunk: {e}")
        return []


async def extract_companies_from_text_async_batch(article_text: str, llm, max_chunk_tokens: int = 1000, batch_size: int = 2) -> List[dict]:
    """
    Extract companies from text using optimized batch processing.
    Each chunk is processed individually but in parallel for better reliability.
    """
    try:
        article_text = ensure_text_is_string(article_text)
        chunk_texts = chunk_text_by_tokens(text=article_text, max_chunk_tokens=max_chunk_tokens, model_encoding="cl100k_base")
        
        # Process chunks in batches for rate limiting, but each chunk gets its own API call
        all_results = []
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i + batch_size]
            batch_results = await process_chunks_in_batch(batch, llm, EXTRACTION_FUNCTION)
            all_results.extend(batch_results)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunk_texts) + batch_size - 1)//batch_size}")
        
        # Merge and deduplicate results
        companies = await merge_chunk_results(all_results)
        logger.info(f"Found {len(companies)} unique companies across all chunks")
        
        return companies

    except Exception as e:
        logger.error(f"Error in company extraction: {e}")
        return []
