import json
import requests

GITHUB_COPILOT_API_URL = "https://api.github.com/copilot/suggestions"
NOTE_FILE = "note2self.json"

def fetch_suggestions_from_copilot(note):
    headers = {
        "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
        "Content-Type": "application/json"
    }
    payload = {
        "note": note,
        "query": "This is a summary of last cycle events. Please can you help me take a look at the repo so we can identify an item for the next incremental improvement?"
    }
    response = requests.post(GITHUB_COPILOT_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch suggestions from GitHub Copilot: {response.status_code}")
        return None

def update_note_with_suggestions(suggestions):
    try:
        with open(NOTE_FILE, 'r') as file:
            note = json.load(file)
    except FileNotFoundError:
        note = {"timestamp": None, "improvement": {}, "assessment": ""}
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from note file: {e}")
        note = {"timestamp": None, "improvement": {}, "assessment": ""}

    note.update(suggestions)

    with open(NOTE_FILE, 'w') as file:
        json.dump(note, file)

def main():
    try:
        with open(NOTE_FILE, 'r') as file:
            note = json.load(file)
    except FileNotFoundError:
        note = {"timestamp": None, "improvement": {}, "assessment": ""}
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from note file: {e}")
        note = {"timestamp": None, "improvement": {}, "assessment": ""}

    suggestions = fetch_suggestions_from_copilot(note)
    if suggestions:
        update_note_with_suggestions(suggestions)
        print("Note updated with suggestions from GitHub Copilot.")
    else:
        print("No suggestions received from GitHub Copilot.")

if __name__ == "__main__":
    main()
