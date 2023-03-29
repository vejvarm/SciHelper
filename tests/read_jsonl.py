from chat2 import load_conversation


if __name__ == "__main__":
    results = [{"i": "2023-30-27_193008_e12851ca6a1945519fd8a4c0febc4593"}, {"id": "ab"}]  # \w errors
    # results = [{"id": "2023-30-27_193008_e12851ca6a1945519fd8a4c0febc4593"}, {"id": "a"}]  # correct
    conversation = load_conversation(results)
    print(len(conversation))
    for turn in conversation:
        print(f"{turn['role']}: {turn['content']}\n")
