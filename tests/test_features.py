import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.generator.summarization import generate_summary, generate_section_summaries
from src.core.generator.comparison import generate_comparison

def _prompt_text(captured_parts):
    """Generation calls pass [prompt_str, PIL.Image, ...]; tests assert on the
    concatenated string parts only."""
    return "\n".join(str(p) for p in captured_parts if isinstance(p, str))

def test_generate_summary_mock(monkeypatch):
    """Verifies that generate_summary builds prompts and delegates successfully."""
    mock_response = "### Core Overview\nThis is a mock summary.\n### Key Findings\n- Contribution A\n### Methodology\nMethod B\n### Conclusion\nTakeaway C"
    
    # Track prompt passed to client
    captured_prompt = None
    
    def mock_generate(prompt, temperature=0.2):
        nonlocal captured_prompt
        captured_prompt = prompt
        return mock_response
        
    monkeypatch.setattr(
        "src.core.generator.summarization.generate_content_with_retry",
        mock_generate
    )
    
    mock_chunks = [
        {"text": "This is page 1 content of research paper.", "source_file": "attention-is-all-you-need-Paper.pdf"},
        {"text": "This is page 2 content explaining Transformer layers.", "source_file": "attention-is-all-you-need-Paper.pdf"}
    ]
    
    summary = generate_summary(mock_chunks)

    # Assertions
    assert summary == mock_response
    prompt_text = _prompt_text(captured_prompt)
    assert "attention-is-all-you-need-Paper.pdf" in prompt_text
    assert "This is page 1 content" in prompt_text
    assert "Transformer layers" in prompt_text

def test_generate_section_summaries_mock(monkeypatch):
    """Verifies that generate_section_summaries groups chunks by section title and requests section-wise summaries."""
    mock_response = "### Introduction\n* Mock introduction summary.\n### Methodology\n* Mock methodology summary."
    
    captured_prompt = None
    
    def mock_generate(prompt, temperature=0.2):
        nonlocal captured_prompt
        captured_prompt = prompt
        return mock_response
        
    monkeypatch.setattr(
        "src.core.generator.summarization.generate_content_with_retry",
        mock_generate
    )
    
    mock_chunks = [
        {"text": "Intro paragraph 1.", "section_title": "Introduction", "source_file": "paper_a.pdf"},
        {"text": "Intro paragraph 2.", "section_title": "Introduction", "source_file": "paper_a.pdf"},
        {"text": "Method details.", "section_title": "Methodology", "source_file": "paper_a.pdf"}
    ]
    
    summary = generate_section_summaries(mock_chunks)
    
    # Assertions
    assert summary == mock_response
    assert "--- SECTION: Introduction ---" in captured_prompt
    assert "--- SECTION: Methodology ---" in captured_prompt
    assert "Intro paragraph 1." in captured_prompt
    assert "Method details." in captured_prompt

def test_generate_comparison_mock(monkeypatch):
    """Verifies that generate_comparison constructs the side-by-side comparison matrix and prompts for both papers."""
    mock_response = "| Dimension | Paper A | Paper B |\n| --- | --- | --- |\n| Core Architecture | Transformers | RNNs |"
    
    captured_prompt = None
    
    def mock_generate(prompt, temperature=0.2):
        nonlocal captured_prompt
        captured_prompt = prompt
        return mock_response
        
    monkeypatch.setattr(
        "src.core.generator.comparison.generate_content_with_retry",
        mock_generate
    )
    
    mock_chunks_a = [{"text": "Self-attention mechanism details.", "source_file": "paper_transformer.pdf"}]
    mock_chunks_b = [{"text": "Long Short-Term Memory details.", "source_file": "paper_lstm.pdf"}]

    docs_data = [
        {"name": "paper_transformer.pdf", "chunks": mock_chunks_a},
        {"name": "paper_lstm.pdf", "chunks": mock_chunks_b}
    ]

    comparison = generate_comparison(docs_data)

    # Assertions
    assert comparison == mock_response
    prompt_text = _prompt_text(captured_prompt)
    assert "Paper A: paper_transformer.pdf" in prompt_text
    assert "Paper B: paper_lstm.pdf" in prompt_text
    assert "Self-attention mechanism" in prompt_text
    assert "Long Short-Term Memory" in prompt_text
