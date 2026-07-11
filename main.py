from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()

def run_pipeline(source :str, language :str = "english", mode: str = "local") -> dict:
    print("starting AI Video Assistant")

    chunks = process_input(source)

    transcript = transcribe_all(chunks, language, mode=mode)
    print(f"raw transcription (first 300 characters ) {transcript[:300]}")

    title = generate_title(transcript, mode=mode)

    summary = summarize(transcript, mode=mode)

    action_item = extract_action_items(transcript, mode=mode)

    decisions = extract_key_decisions(transcript, mode=mode)
    questions = extract_questions(transcript, mode=mode)
    
    rag_chain = build_rag_chain(transcript, mode=mode)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"

    user_mode = input("Mode (local/cloud): ").strip().lower() or "local"
    mode = "api" if user_mode in ("cloud", "api") else "local"
    result = run_pipeline(source, language, mode=mode)
    
    if mode not in ("local", "cloud"):
        mode = "local"
    result = run_pipeline(source, language, mode=mode)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")  
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")