"""
Interactive CLI module for the RAG pipeline.
Allows users to select a manual and query the database using the hybrid retrieval and reranking pipeline.
"""
import os
import sys
from pathlib import Path
import lancedb
import torch
from dotenv import load_dotenv
from answer import load_llm, generate, RETRIEVE_K, RERANK_TOP
from retrieve import init_retriever, init_reranker, retrieve_candidates, rerank_candidates

load_dotenv(Path(__file__).parent.parent / ".env")
if not os.environ.get("HF_HOME"): os.environ["HF_HOME"] = str(Path.home() / "hf_cache")

def fetch_available_manual_names(database_directory_string):
    database_path_object = Path(database_directory_string)
    if not database_path_object.is_dir():
        sys.exit(f"Error: {database_directory_string} directory not found.")
    available_tables_list = [table_directory.name[:-6] for table_directory in database_path_object.iterdir() if table_directory.is_dir() and table_directory.name.endswith(".lance")]
    if not available_tables_list:
        sys.exit(f"Error: No tables found in {database_directory_string}.")
    return sorted(available_tables_list)

def prompt_user_for_manual_selection(available_manuals_list):
    print("\n╔══════════════════════════════════════════════════════════╗\n║                   AVAILABLE MANUALS                      ║\n╚══════════════════════════════════════════════════════════╝")
    for index_number, manual_name_string in enumerate(available_manuals_list, 1):
        print(f"  {index_number}. {manual_name_string}")
    
    while True:
        try:
            user_input_string = input("\nSelect a manual by number (or 'q' to quit): ").strip()
            if not user_input_string: continue
            if user_input_string.lower() in {"exit", "quit", "q", ":q"}: sys.exit(0)
            selected_index_integer = int(user_input_string) - 1
            if 0 <= selected_index_integer < len(available_manuals_list): return available_manuals_list[selected_index_integer]
            print(f"Invalid choice. Please enter a number between 1 and {len(available_manuals_list)}.")
        except ValueError:
            print("Please enter a valid number.")
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nGoodbye.")

def execute_chat_loop(vector_embedder_model, lancedb_database_directory_string, reranker_model_components, causal_language_model, available_manuals_list, initial_manual_name_string):
    current_manual_name_string = initial_manual_name_string
    current_database_connection = lancedb.connect(lancedb_database_directory_string).open_table(current_manual_name_string)
    
    while True:
        display_manual_name_string = current_manual_name_string[:39] + "..." if len(current_manual_name_string) > 42 else current_manual_name_string
        print(f"\n╔══════════════════════════════════════════════════════════╗\n║  Manual Q&A: {display_manual_name_string:<42}  ║\n║  Commands: exit / quit | 'switch' to change manual       ║\n╚══════════════════════════════════════════════════════════╝")
        
        while True:
            try:
                user_query_string = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                sys.exit("\nGoodbye.")
            if not user_query_string: continue
            if user_query_string.lower() in {"exit", "quit", "q", ":q"}: sys.exit("Goodbye.")
            if user_query_string.lower() == "switch":
                current_manual_name_string = prompt_user_for_manual_selection(available_manuals_list)
                current_database_connection = lancedb.connect(lancedb_database_directory_string).open_table(current_manual_name_string)
                break

            print("  Searching...", flush=True)
            vector_embedder_model.cuda()
            initial_retrieved_passages = retrieve_candidates(vector_embedder_model, current_database_connection, user_query_string, k=RETRIEVE_K)
            vector_embedder_model.cpu()
            torch.cuda.empty_cache()
            print("  Reranking...", flush=True)
            final_reranked_passages = rerank_candidates(reranker_model_components, user_query_string, initial_retrieved_passages, top_n=RERANK_TOP)
            print("  Generating...\n", flush=True)
            generate(causal_language_model, user_query_string, final_reranked_passages, stream=True)

if __name__ == "__main__":
    lancedb_database_directory_string = "lancedb"
    available_manuals_list = fetch_available_manual_names(lancedb_database_directory_string)
    selected_manual_name_string = prompt_user_for_manual_selection(available_manuals_list)
    
    print(f"\nLoading models and connecting to '{selected_manual_name_string}'...")
    dense_vector_embedder_model, _ = init_retriever(lancedb_database_directory_string, selected_manual_name_string)
    cross_encoder_reranker_model_components = init_reranker()
    dense_vector_embedder_model.cpu()
    torch.cuda.empty_cache()
    qwen_causal_language_model_components = load_llm()
    
    execute_chat_loop(dense_vector_embedder_model, lancedb_database_directory_string, cross_encoder_reranker_model_components, qwen_causal_language_model_components, available_manuals_list, selected_manual_name_string)
