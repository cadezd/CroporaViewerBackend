import json
import os
import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import spacy


PATH_TO_XML_FILES = "D:\\diplomska-data\\first-parsing\\second-attempt"
PATH_TO_JSONL_FILES = "D:\\diplomska-data\\second-parsing\\"

PATH_TO_ATTENDEES_TEXT_FILE = "../raw-data/seznam_poslancev.txt"
PATH_TO_PLACES_XLSX = "../raw-data/Krajevna_imena.xlsx"
PLACES_SHEET_NAME = "Sheet2"

SKIP_FILES_TO = 0  # set to 0 if you want to convert all files
MAX_FILES = 10  # set to -1 if you want to convert all files
NAMESPACE_MAPPINGS = {"ns0": "http://www.tei-c.org/ns/1.0",
                      "xml": "http://www.w3.org/XML/1998/namespace"}

CORPUS_NAME = "DezelniZborKranjski"

TRANSLATION_API_URL = "http://localhost:5000/translate"

prop_nouns = set()

nlp_sl = spacy.load("sl_core_news_md")
nlp_de = spacy.load("de_core_news_md")


def save_to_jsonl(elements, file_name):
    with open(PATH_TO_JSONL_FILES + "/" + file_name, "w", encoding="utf-8") as file:
        for element in elements:
            file.write(json.dumps(element, ensure_ascii=False) + "\n")
    print("Saved " + str(len(elements)) + " elements to " + file_name)


# parses the attributes of an element
def parse_attribs(attribs):
    parsed_attribs = {}

    for attrib in attribs:
        attrib_name = attrib.split("}")[-1]
        parsed_attribs[attrib_name] = attribs[attrib]

    return parsed_attribs


# parses the tag name from the element
def parse_tag(element):
    tag_name = element.tag.split("}")[-1]
    return tag_name


# parses the date from the meeting id
def parse_date_from_id(id):
    date = id.split("_")[1].split("-")
    return date[2] + "." + date[1] + "." + date[0]


# finds all titles with language attribute and returns them as a list of dictionaries
def parse_titles(xml_root):
    titles = []

    for title in xml_root.findall(".//ns0:title", NAMESPACE_MAPPINGS):
        attribs = parse_attribs(title.attrib)
        if (attribs != {}):
            titles.append({
                "title": title.text,
                "lang": attribs["lang"]
            })

    return titles


# parses the coordinates of an element
def parse_coordinates(element):
    coordinates = []

    try:
        from_page = int(element.get("fromPage"))
        to_page = int(element.get("toPage"))
        x_coords = [float(element.attrib.get(key)) for key in element.attrib.keys() if key.startswith("x")]
        y_coords = [float(element.attrib.get(key)) for key in element.attrib.keys() if key.startswith("y")]
        coords = list(zip(x_coords, y_coords))

        for i in range(0, len(coords), 2):
            rect_coords = coords[i:i + 2]
            x0 = rect_coords[0][0]
            y0 = rect_coords[0][1]
            x1 = rect_coords[1][0]
            y1 = rect_coords[1][1]

            coordinates.append({
                "page": from_page if from_page == to_page else (from_page if i == 0 else to_page),
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1
            })
    except Exception as e:
        # print("parse_coordinates(): error parsing coordinates: " + str(e))
        # print(element.attrib)
        return []

    return coordinates


# finds an element by its id
def find_element_by_id(id, xml_root):
    element = xml_root.find(f'.//ns0:w[@xml:id="{id}"]', NAMESPACE_MAPPINGS)
    if element is None:
        element = xml_root.find(f'.//ns0:s[@xml:id="{id}"]', NAMESPACE_MAPPINGS)
    if element is None:
        element = xml_root.find(f'.//ns0:pc[@xml:id="{id}"]', NAMESPACE_MAPPINGS)

    return element


def translate_text(text, source_lang, target_lang):
    payload = {
        "q": text,
        "source": source_lang,
        "target": target_lang
    }

    response = requests.post(TRANSLATION_API_URL, json=payload)

    if response.status_code == 200:
        return response.json()["translatedText"]
    else:
        print("translate_text(): translation API returned status code " + str(response.status_code))
        return ""


# finds all agendas and contents and returns them as a list of dictionaries
def parse_agendas(xml_root, meeting_id):
    agendas = []

    # find all lists with type="agenda"
    for agenda in xml_root.findall(".//ns0:list[@type='agenda']", NAMESPACE_MAPPINGS):
        # first we handle agendas
        agenda_items = []
        agenda_index = 0
        for item in agenda:
            if parse_tag(item) == "item":
                agenda_index += 1
                agenda_items.append({
                    "n": agenda_index,
                    "text": item.text
                })

        agendas.append(
            {
                "lang": parse_attribs(agenda.attrib)["lang"],
                "items": agenda_items
            }
        )

    # get translations if necessary NOT WORKING
    if len(agendas) == 1:
        if agendas[0]["lang"] == "de":
            print("translating: ", meeting_id, " to sl")
            agenda_sl = []
            for item in agendas[0]["items"]:
                agenda_sl.append({
                    "n": item["n"],
                    "text": translate_text(item["text"], "de", "sl")
                })
            agendas.append({
                "lang": "sl",
                "items": agenda_sl
            })
        elif agendas[0]["lang"] == "sl":
            print("translating: ", meeting_id, " to de")
            agenda_de = []
            for item in agendas[0]["items"]:
                agenda_de.append({
                    "n": item["n"],
                    "text": translate_text(item["text"], "sl", "de")
                })
            agendas.append({
                "lang": "de",
                "items": agenda_de
            })
        else:
            print("parse_agendas(): language '" + agendas[0]["lang"] + "' not supported")
            return []

    return agendas


def lemmanize_text(text, lang, sentence_id):
    if lang == 'sl':
        doc = nlp_sl(text)
    elif lang == 'de':
        doc = nlp_de(text)
    else:
        print("lemmanize_text(): language '" + lang + "' not supported")
        return []

    words = []

    for i, token in enumerate(doc):
        word = {}
        word["id"] = sentence_id + "." + str(token.i + 1) + ".(" + lang + ")"
        word["type"] = "pc" if token.is_punct else "w"
        word["lemma"] = token.lemma_
        word["text"] = token.text
        word["propn"] = 1 if token.pos_ == "PROPN" else 0

        # Adjust join attribute (needed to reconstruct the original text)
        word["join"] = "natural"
        if i < len(doc) - 1 and not token.whitespace_:
            word["join"] = "right"

        words.append(word)

    return words


def parse_sentence(sentence_root, segment_page, segment_id, speaker):
    global prop_nouns
    sentence_attribs = parse_attribs(sentence_root.attrib)
    sentence = {}
    sentence["id"] = sentence_attribs["id"]
    sentence["translations"] = []
    sentence["segment_page"] = segment_page
    sentence["segment_id"] = segment_id
    sentence["speaker"] = speaker

    translation = {}
    translation["lang"] = sentence_attribs["lang"]
    translation["speaker"] = sentence["speaker"]
    translation["original"] = 1
    translation["text"] = ""
    translation["words"] = []

    # parse original language
    for i, word_root in enumerate(sentence_root):
        word_tag = parse_tag(word_root)

        if word_tag == "w" or word_tag == "pc":
            word = {}
            word_attribs = parse_attribs(word_root.attrib)
            word["id"] = word_attribs["id"]
            word["type"] = word_tag
            word["lemma"] = word_attribs["lemma"]
            word["text"] = word_root.text
            word["join"] = word_attribs.get("join", "natural")

            # Determine if the word is a proper noun
            upostag = word_attribs.get("msd").split("|")[0].split("=")[1]
            if upostag == "PROPN":
                word["propn"] = 1
                prop_nouns.add(word["lemma"])
            else:
                word["propn"] = 0


            if not word_tag == "pc" and not word["text"] == "":
                translation["text"] += " "
            translation["text"] += word["text"]

            translation["words"].append(word)

        else:
            print("parse_sentence(): expected child tag 'w' or 'pc', got tag '" + word_tag + "'")
            return False

    sentence["translations"].append(translation)
    sentence["original_language"] = translation["lang"]

    return sentence


def parse_note(note_root, segment_page, segment_id, speaker):
    note = {}
    attribs = parse_attribs(note_root.attrib)

    note["type"] = "comment"
    note["text"] = note_root.text
    note["page"] = segment_page
    note["segment_id"] = segment_id
    note["speaker"] = speaker

    return note


def parse_segment(segment_root, speaker):
    sentences = []
    notes = []
    attribs = parse_attribs(segment_root.attrib)

    segment_page = attribs["n"] if attribs.get("n") else -1
    segment_id = attribs["id"]

    for child in segment_root:
        child_tag = parse_tag(child)
        if child_tag == "s":
            sentences.append(parse_sentence(child, segment_page, segment_id, speaker))
        elif child_tag == "note":
            notes.append(parse_note(child, segment_page, segment_id, speaker))
        else:
            print("parse_segment(): expected child tag 's' or 'note', got tag '" + child_tag + "'")
            return False

    return sentences, notes


def parse_speeches(xml_root):
    sentences = []
    notes = []
    speaker = ""

    # find debateSection
    debate_section = xml_root.find(".//ns0:div[@type='debateSection']", NAMESPACE_MAPPINGS)
    for child in debate_section:
        tag = parse_tag(child)
        attribs = parse_attribs(child.attrib)

        if (tag == "u" or tag == "p") and parse_tag(child[0]) == "seg":
            parsed_sentences, parsed_notes = parse_segment(child[0], speaker)
            sentences.extend(parsed_sentences)
            notes.extend(parsed_notes)
        elif tag == "note" and attribs["type"] == "speaker":
            speaker = re.sub(r'[^a-zA-ZäöüßÄÖÜčšžČŠŽ. 0-9]', '', child.text)
        elif tag == "div":
            for sub_child in child:
                sub_tag = parse_tag(sub_child)
                if (sub_tag == "u" or sub_tag == "p") and parse_tag(sub_child[0]) == "seg":
                    parsed_sentences, parsed_notes = parse_segment(sub_child[0], speaker)
                    sentences.extend(parsed_sentences)
                    notes.extend(parsed_notes)
                elif tag == "note" and attribs["type"] == "speaker":
                    speaker = re.sub(r'[^a-zA-ZäöüßÄÖÜčšžČŠŽ. 0-9]', '', sub_tag.text)
        # else:
        #    print("parse_speeches(): expected child tag 'u' or 'p' or 'note', got tag '" + tag + "'")

    return sentences, notes


def translate_sentences(sentences, ids, source_lang, target_lang):
    translations = []

    payload = {
        "q": sentences,
        "source": source_lang,
        "target": target_lang
    }

    response = requests.post(TRANSLATION_API_URL, json=payload)

    if response.status_code == 200:
        translations = response.json()["translatedText"].split("\n")
    else:
        print("translate_sentences(): translation API returned status code " + str(response.status_code))

    return translations


# translates the sentences and agendas in a meeting
def translate_meeting(meeting):
    start_time = time.time()

    # init lists for sentences
    de_sentence_ids = []
    de_translations_list = []

    sl_sentence_ids = []
    sl_translations_list = []

    # get all sentences in the meeting and put them in lists according to their language, also get their ids
    for sentence in meeting["sentences"]:
        if sentence["translations"][0]["lang"] == "de":
            de_sentence_ids.append(sentence["id"])
            de_translations_list.append(sentence["translations"][0]["text"])
        elif sentence["translations"][0]["lang"] == "sl":
            sl_sentence_ids.append(sentence["id"])
            sl_translations_list.append(sentence["translations"][0]["text"])

    # join the sentences into one string
    de_translations = " \n".join(de_translations_list)
    sl_translations = " \n".join(sl_translations_list)

    # translate german to slovene
    translations = translate_sentences(de_translations, de_sentence_ids, 'de', 'sl')

    mid_time1 = time.time()
    print("translating to de took " + str(mid_time1 - start_time) + " seconds")

    for i in range(len(translations)):
        translation = {}
        translation["lang"] = "sl"
        translation["original"] = 0
        translation["speaker"] = meeting["sentences"][i]["speaker"]
        translation["text"] = translations[i]
        translation["words"] = lemmanize_text(translations[i], 'sl', de_sentence_ids[i])
        sentence_index = next(
            (index for (index, d) in enumerate(meeting["sentences"]) if d["id"] == de_sentence_ids[i]), None)
        meeting["sentences"][sentence_index]["translations"].append(translation)

    mid_time2 = time.time()
    print("lemmatizing de took " + str(mid_time2 - mid_time1) + " seconds")

    # translate slovene to german
    translations = translate_sentences(sl_translations, sl_sentence_ids, 'sl', 'de')

    mid_time3 = time.time()
    print("translating to sl took " + str(mid_time3 - mid_time2) + " seconds")

    for i in range(len(translations)):
        translation = {}
        translation["lang"] = "de"
        translation["original"] = 0
        translation["speaker"] = meeting["sentences"][i]["speaker"]
        translation["text"] = translations[i]
        translation["words"] = lemmanize_text(translations[i], 'de', sl_sentence_ids[i])
        sentence_index = next(
            (index for (index, d) in enumerate(meeting["sentences"]) if d["id"] == sl_sentence_ids[i]), None)
        meeting["sentences"][sentence_index]["translations"].append(translation)

    end_time = time.time()
    print("lemmatizing sl took " + str(end_time - mid_time3) + " seconds")
    print(
        "translate_meeting(): translated " + str(len(de_sentence_ids) + len(sl_sentence_ids)) + " sentences in " + str(
            end_time - start_time) + " seconds")

    # translate agendas

    return


# generates / transforms data for sentences index
def transform_sentences(meeting, xml_root):
    time_start = time.time()

    transformed_sentences = []

    for sentence in meeting["sentences"]:

        # From the sentence element, get the coordinates of the words
        sentence_xml_element = find_element_by_id(sentence["id"], xml_root)
        coordinates = []
        if sentence_xml_element is not None:
            for word_elem in sentence_xml_element.iter():
                word_tag = parse_tag(word_elem)
                if word_tag != "w" and word_tag != "pc":
                    continue
                coordinates.extend(parse_coordinates(word_elem))

        transformed_sentences.append({
            "meeting_id": meeting["id"],
            "sentence_id": sentence["id"],
            "segment_id": sentence["segment_id"],
            "speaker": sentence["speaker"],
            "coordinates": coordinates,
            "translations": [{
                "text": translation["text"],
                "lang": translation["lang"],
                "original": translation["original"]
            } for translation in sentence.get('translations', [])],
        })

    time_end = time.time()

    print("transform_sentences(): transformed sentences in " + str(time_end - time_start) + " seconds")

    return transformed_sentences


# generates / transforms data for words index
def transform_words(meeting, xml_root):
    time_start = time.time()

    transformed_words = []

    for sentence in meeting["sentences"]:
        for translation in sentence["translations"]:

            word_index = 0
            for i, word in enumerate(translation["words"]):

                word_attr = parse_attribs(word)

                # main point of word_index is to "group" punctuations and words together...
                # gives us better accuracy when searching for words in the text in elasticsearch-db search
                word_index = word_index + 1 if i > 0 and translation["words"][i - 1]["join"] != "right" else word_index

                # Get the word element that corresponds to the word id
                word_elem = find_element_by_id(word_attr["id"], xml_root) if translation["original"] == 1 else None
                # From the word element, get the coordinates of the word
                coordinates = []
                if word_elem is not None:
                    coordinates.extend(parse_coordinates(word_elem))

                transformed_words.append({
                    "meeting_id": meeting["id"],
                    "sentence_id": sentence["id"],
                    "segment_id": sentence["segment_id"],
                    "word_id": word_attr["id"],
                    "type": word_attr["type"],
                    "join": word_attr["join"],
                    "text": word_attr["text"],
                    "lemma": word_attr["lemma"],
                    "speaker": sentence["speaker"],
                    "pos": i,  # position of word / punctuation in the sentence
                    # position of word (element which has at least one alphanumeric char) in the sentence (punctuations are part of the word)
                    "wpos": word_index,
                    "coordinates": coordinates,
                    "lang": translation["lang"],
                    "original": translation["original"],
                    "propn": word_attr["propn"]
                })

    time_end = time.time()

    print("transform_words(): transformed words in " + str(time_end - time_start) + " seconds")

    return transformed_words


def parse_zapisnik(xml_root):
    meeting_parse_start_time = time.time()

    meeting = {}

    # get the meeting id
    meeting["id"] = xml_root.attrib["{http://www.w3.org/XML/1998/namespace}id"]

    # get the meeting date
    meeting["date"] = parse_date_from_id(meeting["id"])

    # get the meeting title
    meeting["titles"] = parse_titles(xml_root)

    # get agendas
    meeting["agendas"] = parse_agendas(xml_root, meeting["id"])

    # get speeches
    meeting["sentences"], meeting["notes"] = parse_speeches(xml_root)

    # translate meeting
    translate_meeting(meeting)

    # set corpus
    meeting["corpus"] = CORPUS_NAME

    # generate data for sentences index
    sentences = transform_sentences(meeting, xml_root)

    # generate data for words index
    words = transform_words(meeting, xml_root)

    meeting_parse_end_time = time.time()
    print("parse_zapisnik(): parsed meeting in " + str(meeting_parse_end_time - meeting_parse_start_time) + " seconds")

    # save data to jsonl files
    save_to_jsonl([meeting], meeting["id"] + "_meeting.jsonl")
    save_to_jsonl(sentences, meeting["id"] + "_sentences.jsonl")
    save_to_jsonl(words, meeting["id"] + "_words.jsonl")


def parse_krajevna_imena():
    df = pd.read_excel(PATH_TO_PLACES_XLSX, sheet_name=PLACES_SHEET_NAME)

    slo_names = df.iloc[:, 3].tolist()
    ger_names = df.iloc[:, 4].tolist()

    krajevna_imena = []

    # if there is no translation for a name, we set it to "zzzzz" so that it is sorted to the end of the list
    for noun in prop_nouns:
        if noun in slo_names:
            index = slo_names.index(noun)
            krajevno_ime = {}
            krajevno_ime["corpus"] = CORPUS_NAME
            krajevno_ime["sl"] = noun
            krajevno_ime["de"] = ger_names[index] if isinstance(ger_names[index], str) else "zzzzz"
            krajevna_imena.append(krajevno_ime)
        elif noun in ger_names:
            index = ger_names.index(noun)
            krajevno_ime = {}
            krajevno_ime["corpus"] = CORPUS_NAME
            krajevno_ime["sl"] = slo_names[index] if isinstance(slo_names[index], str) else "zzzzz"
            krajevno_ime["de"] = noun
            krajevna_imena.append(krajevno_ime)

    return krajevna_imena


def parse_poslanci():
    file = open(PATH_TO_ATTENDEES_TEXT_FILE, "r")
    poslanci_lines = file.readlines()
    poslanec_id = 0
    file.close()

    poslanci_list = []
    for line in poslanci_lines:
        line = line.strip()
        if line == "":
            continue

        names = line.split(", ")
        poslanec = {}
        poslanec["id"] = 'posl_' + str(poslanec_id)
        poslanec["names"] = names
        poslanec["corpus"] = CORPUS_NAME
        poslanci_list.append(poslanec)

        poslanec_id += 1

    return poslanci_list


# main function
if __name__ == "__main__":

    # TODO: uncomment this part if you want to parse poslanci and krajevna imena
    # parse_poslanci()
    # parse_krajevna_imena()

    files_converted = 0

    # get a list of all file names in Zapisniki_xml folder
    xml_files = os.listdir(PATH_TO_XML_FILES)
    xml_files = sorted(xml_files)

    # convert xml files to jsonl
    for xml_file in xml_files:
        if MAX_FILES - files_converted == 0:
            break

        if files_converted < SKIP_FILES_TO:
            files_converted += 1
            continue

        if not xml_file.endswith(".xml") or not xml_file.startswith("DezelniZborKranjski"):
            continue

        xml_tree = ET.parse(PATH_TO_XML_FILES + "/" + xml_file)
        xml_root = xml_tree.getroot()

        # initialize parser
        parse_zapisnik(xml_root)

        files_converted += 1
        print("progress: " + str(files_converted) + "/" + str(len(xml_files)), end="\n\n")

    # print("Uploaded meetings and sentences to Elasticsearch")
    print("Saved meetings and sentences to jsonl")
