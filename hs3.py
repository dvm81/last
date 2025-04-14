ef hybrid_search(companies, config: SearchConfig = SearchConfig()):
    """
    Hybrid search implementation that exactly matches the original behavior.
    
    Args:
        companies: List of company dictionaries to search
        config: SearchConfig object with search parameters
        
    Returns:
        DataFrame with search results
    """
    results_list = []

    for company in companies:
        word_orig = company['Word']
        word = word_orig.upper()

        ric_value = (company.get("RIC") or {}).get("value")
        companyRic = ric_value.upper() if isinstance(ric_value, str) else None

        bbticker_value = (company.get("BBTicker") or {}).get("value")
        companyBBTicker = bbticker_value.upper() if isinstance(bbticker_value, str) else None

        symbol_value = (company.get("Symbol") or {}).get("value")
        companySymbol = symbol_value.upper() if isinstance(symbol_value, str) else None

        isin_value = (company.get("ISIN") or {}).get("value")
        companyIsin = isin_value.upper() if isinstance(isin_value, str) else None

        sedol_value = (company.get("SEDOL") or {}).get("value")
        companySedol = sedol_value.upper() if isinstance(sedol_value, str) else None

        # create id list
        identifiers = get_identifiers(company)

        if len(identifiers) == 0:
            # vector search on original word
            issueName_value = company.get("IssueName", {}).get("value", "")
            if issueName_value and issueName_value.casefold() != word_orig.casefold():
                word_vector_results = retrieve_similar_entities(issueName_value, table, 
                                        limit=config.limit_res_vs, threshold=config.sim_threshold)
                if word_vector_results or not any(x.get("IssueName") == word_orig for x in word_vector_results):
                    star_res = query_duckdb_table_case_ins('instruments', 'IssueName', word_orig, 
                                                           top_n=config.limit_results_star).to_dict(orient='records')
                    for record in star_res:
                        record['Word'] = word_orig
                    results_list.extend(star_res)

                for vector_result in word_vector_results:
                    if isinstance(vector_result, dict) and 'IssueName' in vector_result:
                        issue_name = vector_result['IssueName']
                    elif isinstance(vector_result, list) and len(vector_result) > 0 and isinstance(vector_result[0], dict):
                        issue_name = vector_result[0].get('IssueName', None)
                    else:
                        continue

                    if issue_name:
                        star_res = query_duckdb_table_case_ins('instruments', 'IssueName', issue_name, 
                                                               top_n=config.limit_results_star).to_dict(orient='records')
                        for record in star_res:
                            record['Word'] = word_orig
                        results_list.extend(star_res)
            else:
                subresult_list = []

        # 0. Add what the model is most confident about
        if identifiers:
            field_name, value = identifiers[0]
            topid_star_res = query_duckdb_table_case_ins('instruments', field_name, value, 
                                                         top_n=config.limit_results_star).to_dict(orient='records')
            for record in topid_star_res:
                record['Word'] = company['Word']
                results_list.extend(topid_star_res)

        # 1. Direct match between word and same identifier
        direct_match = False
        for field_name, value in identifiers:
            if word_orig.strip().casefold() == value.strip().casefold():
                top_res = query_duckdb_table_case_ins('instruments', field_name, value, 
                                                      top_n=config.limit_results_star).to_dict(orient='records')
                for record in top_res:
                    record['Word'] = company['Word']
                    results_list.extend(top_res)
                direct_match = True
                break

        # 2. Check for partial match
        if not direct_match:
            for field_name, value in identifiers:
                if word_orig.strip().casefold() in value.strip().casefold():
                    records = query_duckdb_table_like_ins('instruments', field_name, value, 
                                                          top_n=config.limit_results_star).to_dict(orient='records')
                    for record in records:
                        record['Word'] = company['Word']
                        results_list.append(record)

        # 3. Speculative matching with limited results
        if not results_list:
            word_vector_results = retrieve_similar_entities(word_orig, table, 
                                                            limit=config.limit_res_vs, threshold=config.sim_threshold)
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
                    query_results = query_duckdb_table_like_ins('instruments', field_name, word, 
                                                                top_n=config.limit_results_star).to_dict(orient='records')
                    for subresult in query_results:
                        subresult['Word'] = company['Word']
                        results_list.append(subresult)
                    break
                elif result.get("RIC") == companyRic or \
                     result.get("BBTicker") == companyBBTicker or \
                     result.get("Symbol") == companySymbol or \
                     result.get("ISIN") == companyIsin:
                    if limit_left > 0:
                        result['Word'] = company['Word']
                        results_list.append(result)
                        limit_left -= 1
                else:
                    speculative_results.append(result)

            if speculative_results:
                results_list.extend(speculative_results[:limit_left])

    if results_list:
        final_results = pd.DataFrame(results_list)
        final_results = final_results.drop_duplicates(inplace=False)
        final_results = final_results.groupby("ISIN", as_index=False).first().sort_values(by=["Word", "AverageVolume"], ascending=[False]*2)
    else:
        final_results = pd.DataFrame()

    return final_results 
