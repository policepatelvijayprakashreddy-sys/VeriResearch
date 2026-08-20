from agents.literature_agent import literature_agent


topic = input("Enter a research topic: ")

result = literature_agent(topic)

print("\n===== LITERATURE RESULTS =====\n")
print(result)