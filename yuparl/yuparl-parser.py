import xml.etree.ElementTree as ET
import time
import json
import os
import re
# import pandas as pd
import requests
import spacy
import cyrtranslit
import sys

# from elasticsearch import Elasticsearch, helpers

SKIP_FILES_TO = 574
MAX_FILES = -1
NAMESPACE_MAPPINGS = {"ns0": "http://www.tei-c.org/ns/1.0",
                      "xml": "http://www.w3.org/XML/1998/namespace"}

PATH_TO_XML_INPUT_FILES = '/home/davidlocal/raw-data/yu1Parl/yu1Parl-xml-enriched'
PATH_TO_JSONL_OUTPUT_FILES = '/home/davidlocal/raw-data/yu1Parl/yu1Parl-json'
# TODO
# PATH_TO_PLACES_FILE = './raw_data/YuParl/places.json'
# PATH_TO_ATTENDEES_FILE = './raw_data/YuParl/attendees.json'

# Files that have 80% or more word elements with missing coordinates
EXCEPTIONS = {
    "19190523-PrivremenoNarodnoPredstavnistvo-24.xml",
    "19320126-NarodnoPretstavnistvo-03.xml",
    "19320129-NarodnaSkupstina-01.xml",
    "19320204-NarodnaSkupstina-04.xml",
    "19320209-NarodnaSkupstina-05.xml",
    "19320224-NarodnaSkupstina-12.xml",
    "19320225-NarodnaSkupstina-13.xml",
    "19320411-NarodnaSkupstina-38.xml",
    "19320805-NarodnaSkupstina-48.xml",
    "19321103-NarodnaSkupstina-02.xml",
    "19321115-Senat-04.xml",
    "19330223-Senat-23.xml",
    "19330301-NarodnaSkupstina-36.xml",
    "19331110-NarodnaSkupstina-02.xml",
    "19331124-NarodnaSkupstina-09.xml",
    "19340306-NarodnaSkupstina-27.xml",
    "19340309-NarodnaSkupstina-30.xml",
    "19340326-Senat-19p1.xml",
    "19340623-Senat-23.xml",
    "19341027-NarodnaSkupstina-02.xml",
    "19350607-Senat-prethodna2.xml",
    "19350727-Senat-06p1.xml",
    "19360306-NarodnaSkupstina-20.xml",
    "19360314-Senat-04.xml",
    "19360322-Senat-08.xml",
    "19360323-Senat-09p1.xml",
    "19360324-Senat-10.xml",
    "19361024-NarodnaSkupstina-02.xml",
    "19361222-NarodnaSkupstina-08.xml",
    "19370201-NarodnaSkupstina-09.xml",
    "19370203-NarodnaSkupstina-11.xml",
    "19370305-NarodnaSkupstina-33.xml",
    "19370306-NarodnaSkupstina-34.xml",
    "19370307-NarodnaSkupstina-35.xml",
    "19370320-Senat-11.xml",
    "19371022-NarodnaSkupstina-02.xml",
    "19380315-NarodnaSkupstina-35.xml",
    "19390117-NarodnaSkupstina-prethodna2.xml",
    "19390217-NarodnaSkupstina-02.xml"
}

MEETINGS_INDEX = 'meetings'
PLACES_INDEX = 'places'
ATTENDEES_INDEX = 'attendees'

CORPUS_NAME = 'Yu1Parl'
TRANSLATION_API_URL = 'http://localhost:5000/translate'
"""
Text is either in Slovene or Serbo-Croatian. We consider that the text is in Croatian, if Serbo-Croatian is
written with latinic characters and in Serbian if it is written in cyrillic. Since Libretranslate
does not support croatian language, we translate it between Slovene to Croatian by using 'sl' and 'sr'.
Croatian is used as a sort of bridge language. If we want to translate between Slovene and Serbian,
which is in cyrillic, we use cyrtranslit to convert the text to latinic and then use Libretranslate
to translate it between Slovene and Croatian.
"""

proper_nouns = set()

nlp_sl = spacy.load('sl_core_news_sm')
nlp_hr = spacy.load('hr_core_news_sm')

headers = {
    'Content-Type': 'application/json',
    'accept': 'application/json'
}


# es = Elasticsearch(hosts=['http://localhost:9200'])

# create indices if necessary
# if not es.indices.exists(index=MEETINGS_INDEX):
#     es.indices.create(index=MEETINGS_INDEX)
# if not es.indices.exists(index=PLACES_INDEX):
#     es.indices.create(index=PLACES_INDEX)
# if not es.indices.exists(index=ATTENDEES_INDEX):
#     es.indices.create(index=ATTENDEES_INDEX)

# def upload_to_es(index, data):
#     actions = []
#     for item in data:
#         actions.append({
#             "_op_type": "index",
#             "_index": index,
#             "_source": item
#         })

#     if len(actions) > 1000:
#         for i in range(0, len(actions), 1000):
#             helpers.bulk(es, actions[i:i+1000])
#             print(f"Uploaded {i+1000} items to {index}")
#     else:
#         helpers.bulk(es, actions)
#         print(f"Uploaded {len(actions)} items to {index}")

def save_to_jsonl(filename, data):
    with open(PATH_TO_JSONL_OUTPUT_FILES + '/' + filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

        f.close()
    print(f"Saved {len(data)} items to {filename}")


def parse_attribs(elem):
    attribs = {}
    for key in elem:
        attrib_name = key.split('}')[-1]
        attribs[attrib_name] = elem[key]
    return attribs


def parse_tag(elem):
    return elem.tag.split('}')[-1]


def parse_date_from_id(id):
    date = id.split('_')[1].split('-')
    return f"{date[2]}.{date[1]}.{date[0]}"


def parse_titles(xml_root):
    titles = []
    for title in xml_root.findall('.//ns0:title', NAMESPACE_MAPPINGS):
        attribs = parse_attribs(title.attrib)
        if attribs != {} and 'lang' in attribs:
            titles.append({
                'title': title.text,
                'lang': attribs['lang']
            })

    return titles


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

    response = requests.post(
        TRANSLATION_API_URL,
        json=payload
    )

    if response.status_code == 200:
        return response.json()['translatedText']
    else:
        # throw exception if translation failed
        raise Exception(f"Error translating text: {response}")


def parse_agendas(xml_root):
    agendas = []
    for agenda in xml_root.findall('.//ns0:preAgenda', NAMESPACE_MAPPINGS):
        agenda_attribs = parse_attribs(agenda.attrib)
        if not 'lang' in agenda_attribs or not agenda_attribs['lang'] in ['sl', 'hr', 'sr']:
            print(f"Invalid language: {agenda_attribs['lang']}")
            continue
        agenda_items = []
        agenda_index = 0
        for item in agenda:
            item_attribs = parse_attribs(item.attrib)
            if parse_tag(item) == 'item' and item_attribs['n']:
                agenda_index += 1
                agenda_items.append({
                    'n': agenda_index,
                    'text': item.text
                })

        agendas.append({
            'lang': agenda_attribs['lang'],
            'items': agenda_items
        })

    # translate if necessary
    if len(agendas) == 1:
        if agendas[0]['lang'] == 'sl':
            agenda_hr = []
            agenda_sr = []
            for item in agendas[0]["items"]:
                serbo_croatian_latinic = translate_text(item['text'], 'sl', 'sr')
                serbo_croatian_cyrilic = cyrtranslit.to_cyrillic(serbo_croatian_latinic)
                agenda_hr.append({
                    'n': item['n'],
                    'text': serbo_croatian_latinic
                })
                agenda_sr.append({
                    'n': item['n'],
                    'text': serbo_croatian_cyrilic
                })

            agendas.append({
                'lang': 'hr',
                'items': agenda_hr
            })
            agendas.append({
                'lang': 'sr',
                'items': agenda_sr
            })
        elif agendas[0]['lang'] == 'hr':
            agenda_sl = []
            agenda_sr = []
            for item in agendas[0]["items"]:
                serbo_croatian_cyrilic = cyrtranslit.to_cyrillic(item['text'])
                slovene = translate_text(item['text'], 'sr', 'sl')
                agenda_sl.append({
                    'n': item['n'],
                    'text': slovene
                })
                agenda_sr.append({
                    'n': item['n'],
                    'text': serbo_croatian_cyrilic
                })

            agendas.append({
                'lang': 'sl',
                'items': agenda_sl
            })
            agendas.append({
                'lang': 'sr',
                'items': agenda_sr
            })
        else:
            if not agendas[0]['lang'] == 'sr':
                print(f"Invalid language: {agendas[0]['lang']}")
            agenda_sl = []
            agenda_hr = []
            for item in agendas[0]["items"]:
                serbo_croatian_latinic = cyrtranslit.to_latin(item['text'])
                slovene = translate_text(serbo_croatian_latinic, 'sr', 'sl')
                agenda_sl.append({
                    'n': item['n'],
                    'text': slovene
                })
                agenda_hr.append({
                    'n': item['n'],
                    'text': serbo_croatian_latinic
                })

            agendas.append({
                'lang': 'sl',
                'items': agenda_sl
            })
            agendas.append({
                'lang': 'hr',
                'items': agenda_hr
            })
    elif len(agendas) == 2:
        hr_agenda = next((agenda for agenda in agendas if agenda['lang'] == 'hr'), None)
        sr_agenda = next((agenda for agenda in agendas if agenda['lang'] == 'sr'), None)
        sl_agenda = next((agenda for agenda in agendas if agenda['lang'] == 'sl'), None)

        if hr_agenda is not None and sr_agenda is not None and sl_agenda is None:
            sl_agenda = []
            for item in hr_agenda['items']:
                slovene = translate_text(item['text'], 'sr', 'sl')
                sl_agenda.append({
                    'n': item['n'],
                    'text': slovene
                })

            agendas.append({
                'lang': 'sl',
                'items': sl_agenda
            })

        elif hr_agenda is not None and sl_agenda is not None and sr_agenda is None:
            sr_agenda = []
            for item in hr_agenda['items']:
                serbo_croatian_cyrilic = cyrtranslit.to_cyrillic(item['text'])
                sr_agenda.append({
                    'n': item['n'],
                    'text': serbo_croatian_cyrilic
                })

            agendas.append({
                'lang': 'sr',
                'items': sr_agenda
            })

        elif sl_agenda is not None and sr_agenda is not None and hr_agenda is None:
            hr_agenda = []
            for item in sl_agenda['items']:
                serbo_croatian_latinic = cyrtranslit.to_latin(item['text'])
                hr_agenda.append({
                    'n': item['n'],
                    'text': serbo_croatian_latinic
                })

            agendas.append({
                'lang': 'hr',
                'items': hr_agenda
            })

        else:
            raise Exception(f"Invalid number of agendas is None:\n{hr_agenda}\n{sl_agenda}\n{sr_agenda}")
    else:
        print(f"Found {len(agendas)} agendas. Skipping translation.")

    return agendas


def lemmatize_text(text, lang, sentence_id):
    global proper_nouns

    doc = None
    if lang == 'sl':
        doc = nlp_sl(text)
    elif lang == 'hr' or lang == 'sr':
        doc = nlp_hr(text)
    else:
        raise Exception(f"Invalid language: {lang}")

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

        if token.pos_ == 'PROPN':
            proper_nouns.add(token.lemma_)

    return words


def parse_sentence(sentence_root, segment_page, segment_id, speaker):
    global proper_nouns
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
                proper_nouns.add(word["lemma"])
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
    note = {
        'type': 'comment',
        'text': note_root.text,
        'page': segment_page,
        'segment_id': segment_id,
        'speaker': speaker
    }

    return note


def parse_segment(segment_root, speaker):
    sentences = []
    notes = []
    attribs = parse_attribs(segment_root.attrib)

    segment_page = -1  # no data in xml
    segment_id = attribs['id']

    for child in segment_root:
        child_tag = parse_tag(child)
        if child_tag == 's':
            sentences.append(parse_sentence(child, segment_page, segment_id, speaker))
        elif child_tag == 'note':
            notes.append(parse_note(child, segment_page, segment_id, speaker))
        else:
            print(f"Invalid segment child tag: {child_tag} Skipping.")

    return sentences, notes


def parse_speeches(xml_root):
    sentences = []
    notes = []
    speaker = ""

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

    return sentences, notes


def translate_sentences(sentences, source_lang, target_lang):
    translations = []

    payload = {
        "q": sentences,
        "source": source_lang,
        "target": target_lang
    }

    response = requests.post(
        TRANSLATION_API_URL,
        json=payload
    )

    if response.status_code == 200:
        translations = response.json()['translatedText'].split("\n")
    else:
        raise Exception(f"Error translating text: {response.text}")

    return translations

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
        print("parse_coordinates(): error parsing coordinates: " + str(e))
        return []

    return coordinates


def translate_meeting(meeting):
    start_time = time.time()

    # sentences list
    hr_ids = []
    hr_sentences = []

    sr_ids = []
    sr_sentences = []

    sl_ids = []
    sl_sentences = []

    # filter out none sentences
    meeting['sentences'] = [sentence for sentence in meeting['sentences'] if sentence is not None]

    for sentence in meeting['sentences']:
        if sentence['original_language'] == 'hr':
            hr_ids.append(sentence['id'])
            hr_sentences.append(sentence['translations'][0]['text'])
        elif sentence['original_language'] == 'sr':
            sr_ids.append(sentence['id'])
            sr_sentences.append(sentence['translations'][0]['text'])
        elif sentence['original_language'] == 'sl':
            sl_ids.append(sentence['id'])
            sl_sentences.append(sentence['translations'][0]['text'])
        else:
            print(f"Invalid language: {sentence['original_language']}")
            continue

    hr_translations = '\n'.join(hr_sentences)
    sr_translations = '\n'.join(sr_sentences)
    sl_translations = '\n'.join(sl_sentences)

    if len(hr_translations) > 0:
        # translate hr sentences
        hr2sl = translate_sentences(hr_translations, 'sr', 'sl')
        hr2sr = cyrtranslit.to_cyrillic(hr_translations).split("\n")

        for i in range(len(hr_ids)):
            translation1 = {
                'lang': 'sl',
                'original': 0,
                'text': hr2sl[i],
                'words': lemmatize_text(hr2sl[i], 'sl', hr_ids[i])
            }

            translation2 = {
                'lang': 'sr',
                'original': 0,
                'text': hr2sr[i],
                'words': lemmatize_text(hr2sr[i], 'sr', hr_ids[i])
            }

            sentence_index = next((index for (index, d) in enumerate(meeting['sentences']) if d['id'] == hr_ids[i]),
                                  None)
            meeting['sentences'][sentence_index]['translations'].append(translation1)
            meeting['sentences'][sentence_index]['translations'].append(translation2)

    hr_time = time.time()
    print(f"Translating HR to SL & SR sentences in {hr_time - start_time} seconds")

    if len(sr_translations) > 0:
        # translate sr sentences
        sr_latinic = cyrtranslit.to_latin(sr_translations)
        sr2hr = sr_latinic.split("\n")
        sr2sl = translate_sentences(sr_translations, 'sr', 'sl')

        for i in range(len(sr_ids)):
            translation1 = {
                'lang': 'hr',
                'original': 0,
                'text': sr2hr[i],
                'words': lemmatize_text(sr2hr[i], 'hr', sr_ids[i])
            }

            translation2 = {
                'lang': 'sl',
                'original': 0,
                'text': sr2sl[i],
                'words': lemmatize_text(sr2sl[i], 'sl', sr_ids[i])
            }

            sentence_index = next((index for (index, d) in enumerate(meeting['sentences']) if d['id'] == sr_ids[i]),
                                  None)
            meeting['sentences'][sentence_index]['translations'].append(translation1)
            meeting['sentences'][sentence_index]['translations'].append(translation2)

    sr_time = time.time()
    print(f"Translating SR to HR & SL sentences in {sr_time - hr_time} seconds")

    # translate sl sentences
    if len(sl_translations) > 0:
        sl2hr = translate_sentences(sl_translations, 'sl', 'sr')
        # latinic sentences are already split, so we use to_cyrillic on each sentence
        sl2sr = [cyrtranslit.to_cyrillic(sentence) for sentence in sl2hr]

        for i in range(len(sl_ids)):
            translation1 = {
                'lang': 'hr',
                'original': 0,
                'text': sl2hr[i],
                'words': lemmatize_text(sl2hr[i], 'hr', sl_ids[i])
            }

            translation2 = {
                'lang': 'sr',
                'original': 0,
                'text': sl2sr[i],
                'words': lemmatize_text(sl2sr[i], 'sr', sl_ids[i])
            }

            sentence_index = next((index for (index, d) in enumerate(meeting['sentences']) if d['id'] == sl_ids[i]),
                                  None)
            meeting['sentences'][sentence_index]['translations'].append(translation1)
            meeting['sentences'][sentence_index]['translations'].append(translation2)

    end_time = time.time()
    print(f"Translating SL to HR & SR sentences in {end_time - sr_time} seconds")
    print(f"Translated meeting in {end_time - start_time} seconds")

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
    start_time = time.time()
    meeting_id = xml_root.attrib['{http://www.w3.org/XML/1998/namespace}id']
    sentences, notes = parse_speeches(xml_root)

    meeting = {
        'id': meeting_id,
        'date': parse_date_from_id(meeting_id),
        'titles': parse_titles(xml_root),
        'agendas': parse_agendas(xml_root),
        'sentences': sentences,
        'notes': notes,
        'corpus': CORPUS_NAME
    }

    mid_time = time.time()
    print(f"Parsed meeting in {mid_time - start_time} seconds")

    translate_meeting(meeting)

    # generate data for sentences index
    sentences = transform_sentences(meeting, xml_root)

    # generate data for words index
    words = transform_words(meeting, xml_root)

    save_to_jsonl(f"{meeting_id}.jsonl", [meeting])
    save_to_jsonl(meeting["id"] + "_sentences.jsonl", sentences)
    save_to_jsonl(meeting["id"] + "_words.jsonl", words)
    return meeting


if __name__ == '__main__':
    file_count = 0
    start_time = time.time()
    filelist = list(set(os.listdir(PATH_TO_XML_INPUT_FILES)) - EXCEPTIONS)
    filelist = sorted(filelist, key=lambda x: os.path.getsize(os.path.join(PATH_TO_XML_INPUT_FILES, x)))

    for filename in filelist:
        if file_count < SKIP_FILES_TO:
            file_count += 1
            continue

        if MAX_FILES != -1 and file_count >= MAX_FILES:
            break

        if not filename.endswith('.xml') or not filename.startswith('1'):
            continue

        print("Processing file: " + filename)

        tree = ET.parse(PATH_TO_XML_INPUT_FILES + '/' + filename)
        root = tree.getroot()

        meeting = parse_zapisnik(root)

        file_count += 1
        print(f"Processed {file_count} / {MAX_FILES if MAX_FILES > -1 else len(filelist)} files")
        print(f"Time elapsed: {(time.time() - start_time) / 3600}h\n")


