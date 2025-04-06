"""
Prompts for GPT-4o evaluation of document relevance
"""

def get_passage_relevance_prompt(query, retrieved_passage, gold_passage):
    """
    Returns a prompt for evaluating the relevance of a single passage to a query.
    
    Args:
        query: The search query
        retrieved_passage: The passage to evaluate
        gold_passage: A known relevant passage for comparison
        
    Returns:
        str: The formatted prompt
    """
    return f"""
    You are an expert evaluator for a document retrieval system. Your task is to assess how relevant a passage is to a query.

    Query: {query}

    Retrieved Passage: {retrieved_passage}

    Gold Standard Passage (known to be relevant): {gold_passage}

    Please evaluate:
    1. Relevance Score: Rate the retrieved passage on a scale of 0-5, where:
       - 0: Not relevant at all
       - 1: Tangentially relevant (mentions some keywords but misses the intent)
       - 2: Partially relevant (addresses some aspects of the query)
       - 3: Moderately relevant (addresses the main intent but missing details)
       - 4: Very relevant (addresses the query comprehensively)
       - 5: Perfectly relevant (addresses the query as well as or better than the gold standard)

    2. Justification: Explain your reasoning in 2-3 sentences.

    3. Comparison: Briefly compare how the retrieved passage differs from the gold standard.

    Output your assessment in JSON format:
    {{
      "relevance_score": <score>,
      "justification": "<your explanation>",
      "comparison": "<your comparison>"
    }}
    """

def get_passage_ranking_prompt(query, passages):
    """
    Returns a prompt for ranking multiple passages based on relevance to a query.
    
    Args:
        query: The search query
        passages: List of (passage_num, passage_text) tuples
        
    Returns:
        str: The formatted prompt
    """
    passages_text = "\n\n".join([f"Passage {i+1}: {passage}" for i, passage in enumerate(passages)])
    
    return f"""
    You are an expert evaluator for a document retrieval system. Your task is to rank a set of passages based on their relevance to a query.

    Query: {query}

    {passages_text}

    Please rank these passages from most relevant (1) to least relevant ({len(passages)}). 
    Your ranking should be based on how well each passage addresses the information need in the query.

    Output your ranking as a JSON array of integers representing the passage numbers in order from most to least relevant:
    {{
      "ranking": [most_relevant_passage_num, second_most_relevant_passage_num, ..., least_relevant_passage_num]
    }}
    """