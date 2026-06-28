import os

def detect_summary_request(prompt, indexed_files):
    """
    Parses the prompt to see if it's a request to summarize a document.
    Returns (target_filename, is_section_wise) if matched, otherwise (None, False).
    """
    p_lower = prompt.lower()
    keywords = ["summarize", "summary", "summarization", "executive summary", "summarise"]
    
    # Check if any summary keywords are present
    has_keyword = any(kw in p_lower for kw in keywords)
    if not has_keyword:
        return None, False
        
    if not indexed_files:
        return None, False
        
    # Check if section-wise is requested
    section_keywords = ["section summary", "section-wise", "section wise", "detailed summary"]
    is_section_wise = any(kw in p_lower for kw in section_keywords)
        
    # Attempt 1: Look for exact or partial matches of indexed filenames in the prompt
    for filename in indexed_files:
        name_only = os.path.splitext(filename)[0].lower()
        if filename.lower() in p_lower or name_only in p_lower:
            return filename, is_section_wise
            
    # Attempt 2: Default to the most recently indexed file if keywords exist but no name matches
    return indexed_files[-1], is_section_wise

def detect_comparison_request(prompt, indexed_files):
    """
    Parses the prompt to see if it's a comparison request between multiple documents.
    Returns a list of matched filenames, otherwise an empty list.
    """
    p_lower = prompt.lower()
    keywords = ["compare", "contrast", "difference between", "comparison"]
    has_keyword = any(kw in p_lower for kw in keywords)
    if not has_keyword:
        return []
        
    matched_files = []
    for filename in indexed_files:
        name_only = os.path.splitext(filename)[0].lower()
        if filename.lower() in p_lower or name_only in p_lower:
            matched_files.append(filename)
            
    # Remove duplicates preserving order
    unique_matches = []
    for f in matched_files:
        if f not in unique_matches:
            unique_matches.append(f)
            
    return unique_matches
