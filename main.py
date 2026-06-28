import os
import argparse
import subprocess

def run_ingest(file_path, strategy):
    from src.core.ingestion import parse_pdf
    from src.core.chunker import process_chunks
    from src.core.vector_store import add_document_to_store
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
        
    try:
        elements = parse_pdf(file_path, strategy)
        processed_chunks = process_chunks(elements, file_path)
        add_document_to_store(processed_chunks)
        print("Ingestion completed successfully!")
    except Exception as e:
        print(f"Ingestion failed: {e}")

def run_query(query):
    from src.core.vector_store import search_store
    from src.core.generator import generate_answer
    
    try:
        results = search_store(query, k=5)
        if not results:
            print("No matching context found. Vector database might be empty.")
            return
            
        answer = generate_answer(query, results)
        print("\n=== GEMINI ANSWER ===")
        print(answer)
        print("=====================\n")
    except Exception as e:
        print(f"Query execution failed: {e}")

def run_summarize(doc_name):
    from src.core.vector_store import get_registry
    from src.core.generator import generate_summary
    
    try:
        registry = get_registry()
        doc_chunks = [chunk for chunk in registry if os.path.basename(chunk["source_file"]) == os.path.basename(doc_name)]
        
        if not doc_chunks:
            print(f"Error: No chunks found for document '{doc_name}'. Please ensure it is indexed.")
            return
            
        summary = generate_summary(doc_chunks)
        print(f"\n=== SUMMARY FOR {doc_name} ===")
        print(summary)
        print("=============================\n")
    except Exception as e:
        print(f"Summary generation failed: {e}")

def run_chat():
    from src.core.vector_store import search_store
    from src.core.generator import generate_answer
    
    print("====================================================")
    print("💬 RAG Assistant CLI Chat Mode (type 'exit' to quit)")
    print("====================================================")
    
    chat_history = []
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit", "q"]:
                print("Chat ended.")
                break
                
            if not query.strip():
                continue
                
            results = search_store(query, k=5)
            answer = generate_answer(query, results, chat_history)
            
            print(f"\nAssistant: {answer}")
            
            # Maintain chat history
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})
        except KeyboardInterrupt:
            print("\nChat ended.")
            break
        except Exception as e:
            print(f"\nError: {e}")

def run_ui():
    print("Starting Streamlit Dashboard...")
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "ui.py")
    subprocess.run(["streamlit", "run", ui_path])

def main():
    parser = argparse.ArgumentParser(description="📚 Production Modular Multi-PDF RAG Assistant CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # Ingest subcommand
    parser_ingest = subparsers.add_parser("ingest", help="Ingest and index a PDF file")
    parser_ingest.add_argument("file_path", type=str, help="Path to the PDF file")
    parser_ingest.add_argument("--strategy", type=str, choices=["fast", "hi_res"], default="fast", 
                               help="Parsing strategy (fast or hi_res)")
                               
    # Query subcommand
    parser_query = subparsers.add_parser("query", help="Query the knowledge base for a quick answer")
    parser_query.add_argument("query_text", type=str, help="Your search query")
    
    # Summarize subcommand
    parser_sum = subparsers.add_parser("summarize", help="Generate summary for an indexed document")
    parser_sum.add_argument("doc_name", type=str, help="Filename of the indexed document")
    
    # Chat subcommand
    subparsers.add_parser("chat", help="Start an interactive chat session in CLI")
    
    # UI subcommand
    subparsers.add_parser("ui", help="Launch the Streamlit web dashboard")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        run_ingest(args.file_path, args.strategy)
    elif args.command == "query":
        run_query(args.query_text)
    elif args.command == "summarize":
        run_summarize(args.doc_name)
    elif args.command == "chat":
        run_chat()
    elif args.command == "ui":
        run_ui()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
