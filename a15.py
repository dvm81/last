async def extract_companies_from_text_async_batch(article_text: str, llm, max_chunk_tokens: int = 1000, batch_size: int = 2) -> List[dict]:
    try:
        article_text = ensure_text_is_string(article_text)
        chunk_texts = chunk_text_by_tokens(text=article_text, max_chunk_tokens=max_chunk_tokens, model_encoding="cl100k_base")

        # Process chunks in batches
        all_results = []
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i: i + batch_size]
            try:
                batch_results = await process_chunks_in_batch(batch, llm, EXTRACTION_FUNCTION)
                all_results.extend(batch_results)
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size}: {e}")
                # Continue with next batch instead of returning empty results
        
        # Flatten the list of results
        chunk_results = [company for batch in all_results for company in batch]
        companies = await merge_chunk_results(chunk_results)
        logger.info(f"Found {len(companies)} unique companies across all chunks")

        return companies

    except Exception as e:
        logger.error(f"Error in company extraction: {e}")
        return []


async def process_chunks_in_batch(chunks: List[str], llm, function_schema: dict) -> List[List[dict]]:
    """
    Process multiple chunks in parallel but with separate API calls.
    """
    try:
        # Create separate tasks for each chunk
        tasks = []
        for chunk in chunks:
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
            
            # Create a task for each chunk
            task = llm.chat.completions.create(
                model=model_name,
                messages=messages,
                functions=[function_schema],
                function_call={"name": function_schema["name"]},
                temperature=0.0
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        responses = await asyncio.gather(*tasks)
        
        # Parse the responses
        results = []
        for response in responses:
            try:
                function_args = json.loads(response.choices[0].message.function_call.arguments)
                results.append(function_args.get("companies", []))
            except Exception as e:
                logger.error(f"Error parsing response: {e}")
                results.append([])  # Return empty result for this chunk only
        
        return results

    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return [[] for _ in chunks]  # Return empty results for each chunk on error
