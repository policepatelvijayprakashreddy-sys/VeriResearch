from orchestrator.orchestrator import run_research


topic = input("Enter a research topic: ")

results = run_research(topic)

print("\n\n")
print("=" * 60)
print("                  FINAL RESEARCH REPORT")
print("=" * 60)

print(results["final_report"])