from music21 import environment, converter, note
import json
from icecream import ic

environment.set('musicxmlPath', '/Applications/MuseScore 4.app/Contents/MacOS/mscore')
environment.set('musescoreDirectPNGPath', '/Applications/MuseScore 4.app/Contents/MacOS/mscore')

with open("data/tunes.json") as f:
    tunes = json.load(f)

for tune in tunes:
    abc_header = f"X: {tune['tune_id']}\nT: {tune['name']}\nM: {tune['meter']}\nK: {tune['mode']}\n"
    abc_data = tune["abc"].replace("\\r\\n", "\n")
    complete_abc = abc_header + abc_data

    score = converter.parse(complete_abc)

    sequence = []

    for item in score.recurse().getElementsByClass(['Note', 'Rest']):
        item_type = 'Note' if isinstance(item, note.Note) else 'Rest'
        duration = item.duration.quarterLength
        sequence.append((item_type, duration))
    print(sequence)
    break
