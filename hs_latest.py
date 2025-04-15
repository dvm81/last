def hybrid_search(companies, config: SearchConfig = SearchConfig()):
    """
    Optimized hybrid search implementation that reduces redundant queries and implements early termination.
    
    Args:
        companies: List of company dictionaries to search
        config: SearchConfig object with search parameters
        
    Returns:
        DataFrame with search results
    """
    results_list = []
    processed_companies = set()  # Track processed companies to avoid duplicates

    for company in companies:
        word_orig = company['Word']
        word = word_orig.upper()
        
        # Skip if we've already processed this company
        company_key = f"{word_orig}_{company.get('RIC', {}).get('value', '')}_{company.get('ISIN', {}).get('value', '')}"
        if company_key in processed_companies:
            continue
        processed_companies.add(company_key)

        # Extract identifiers once
        identifiers = get_identifiers(company)
        
        # Early termination if we've found enough results
        if len(results_list) >= config.limit_results_star * len(companies):
            break

        # Process company based on whether it has identifiers
        if not identifiers:
            # No identifiers case - use vector search on original word
            issueName_value = company.get("IssueName", {}).get("value", "")
            
            if issueName_value and issueName_value.casefold() != word_orig.casefold():
                # Use cached vector search
                word_vector_results = cached_retrieve_similar_entities(
                    issueName_value, 
                    table, 
                    limit=config.limit_res_vs, 
                    threshold=config.sim_threshold
                )
                
                # Check if we need to search for the original word
                if not word_vector_results or not any(x.get("IssueName") == word_orig for x in word_vector_results):
                    # Use cached database query
                    star_res = cached_query_duckdb_table_case_ins(
                        'instruments', 
                        'IssueName', 
                        word_orig, 
                        top_n=config.limit_results_star
                    ).to_dict(orient='records')
                    
                    for record in star_res:
                        record['Word'] = word_orig
                    results_list.extend(star_res)

                # Process vector results
                for vector_result in word_vector_results:
                    if isinstance(vector_result, dict) and 'IssueName' in vector_result:
                        issue_name = vector_result['IssueName']
                    elif isinstance(vector_result, list) and len(vector_result) > 0 and isinstance(vector_result[0], dict):
                        issue_name = vector_result[0].get('IssueName', None)
                    else:
                        continue

                    if issue_name:
                        # Use cached database query
                        star_res = cached_query_duckdb_table_case_ins(
                            'instruments', 
                            'IssueName', 
                            issue_name, 
                            top_n=config.limit_results_star
                        ).to_dict(orient='records')
                        
                        for record in star_res:
                            record['Word'] = word_orig
                        results_list.extend(star_res)
        else:
            # Process companies with identifiers
            # 1. Start with most confident identifier
            field_name, value = identifiers[0]
            
            # Use cached database query
            topid_star_res = cached_query_duckdb_table_case_ins(
                'instruments', 
                field_name, 
                value, 
                top_n=config.limit_results_star
            ).to_dict(orient='records')
            
            for record in topid_star_res:
                record['Word'] = company['Word']
            results_list.extend(topid_star_res)

            # 2. Check for direct match between word and identifier
            direct_match = False
            for field_name, value in identifiers:
                if word_orig.strip().casefold() == value.strip().casefold():
                    # Use cached database query
                    top_res = cached_query_duckdb_table_case_ins(
                        'instruments', 
                        field_name, 
                        value, 
                        top_n=config.limit_results_star
                    ).to_dict(orient='records')
                    
                    for record in top_res:
                        record['Word'] = company['Word']
                    results_list.extend(top_res)
                    direct_match = True
                    break

            # 3. Check for partial match if no direct match found
            if not direct_match:
                for field_name, value in identifiers:
                    if word_orig.strip().casefold() in value.strip().casefold():
                        # Use cached database query
                        records = cached_query_duckdb_table_like_ins(
                            'instruments', 
                            field_name, 
                            value, 
                            top_n=config.limit_results_star
                        ).to_dict(orient='records')
                        
                        for record in records:
                            record['Word'] = company['Word']
                        results_list.append(record)

        # 4. Speculative matching with limited results if no results found yet
        if not results_list:
            # Use cached vector search
            word_vector_results = cached_retrieve_similar_entities(
                word_orig, 
                table, 
                limit=config.limit_res_vs, 
                threshold=config.sim_threshold
            )
            
            limit_left = config.limit_spec_results
            speculative_results = []

            for result in word_vector_results:
                if word.casefold() == result.get("RIC", "").casefold() or \
                   word.casefold() == result.get("BBTicker", "").casefold() or \
                   word.casefold() == result.get("ISIN", "").casefold() or \
                   word.casefold() == result.get("SEDOL", "").casefold():
                    result['Word'] = company['Word']
                    results_list.append(result)
                    break
                elif word.casefold() == result.get("Symbol", "").casefold():
                    field_name = "Symbol"
                    # Use cached database query
                    query_results = cached_query_duckdb_table_like_ins(
                        'instruments', 
                        field_name, 
                        word, 
                        top_n=config.limit_results_star
                    ).to_dict(orient='records')
                    
                    for subresult in query_results:
                        subresult['Word'] = company['Word']
                    results_list.append(subresult)
                    break
                elif result.get("RIC") == company.get("RIC", {}).get("value", "").upper() or \
                     result.get("BBTicker") == company.get("BBTicker", {}).get("value", "").upper() or \
                     result.get("Symbol") == company.get("Symbol", {}).get("value", "").upper() or \
                     result.get("ISIN") == company.get("ISIN", {}).get("value", "").upper():
                    if limit_left > 0:
                        result['Word'] = company['Word']
                        results_list.append(result)
                        limit_left -= 1
                else:
                    speculative_results.append(result)

            if speculative_results:
                results_list.extend(speculative_results[:limit_left])

    # Process final results
    if results_list:
        final_results = pd.DataFrame(results_list)
        final_results = final_results.drop_duplicates(inplace=False)
        final_results = final_results.groupby("ISIN", as_index=False).first().sort_values(by=["Word", "AverageVolume"], ascending=[False]*2)
    else:
        final_results = pd.DataFrame()

    return final_results
