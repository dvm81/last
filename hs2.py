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
        
        # Try partial match if no exact matches found
        if not any(any(id[1] == str(r[field]).upper() for id in company['identifiers'] if id[0] == field) 
                 for company in field_companies for r in results_list):
            partial_results = query_duckdb_table_like_ins('instruments', field, value, top_n=config.limit_results_star)
    
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
