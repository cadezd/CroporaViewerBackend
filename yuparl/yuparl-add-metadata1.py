import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from typing import Callable

import edlib
from cyrtranslit import to_latin, to_cyrillic

import pdfplumber
from goto import with_goto
from rich import segment

PATH_TO_XML_FILES = "/home/davidlocal/raw-data/yu1Parl.TEI.ana"
PATH_TO_PDF_FILES = "/home/davidlocal/raw-data/yu1Parl-source"
PATH_TO_WORD_FILES = "/home/davidlocal/raw-data/yu1Parl-source"
OUTPUT_FILE = "/home/davidlocal/raw-data/yuparl-xml-enriched"

# Set to True if you want to visualize the coordinates on the PDF and save the images into a folder
VISUALIZE_COORDINATES_FROM_XML = True
VISUALIZATION_FILE = "/home/davidlocal/raw-data/yuparl-visualizations"

SKIP_FILES_TO = 19  # set to 0 if you want to convert all files
MAX_FILES = 20  # set to -1 if you want to convert all files

# If you want to see alignment for each word in the sentence set this to True
# Target -> word from the xml; Best match -> word from the pdf; Similarity -> similarity between the two words
PRINT_ALIGNMENT = True

# Namespace
TEI = "http://www.tei-c.org/ns/1.0"
NAMESPACE = "http://www.w3.org/XML/1998/namespace"

# Tags in the XML files
SEGMENT_TAG = "{" + TEI + "}seg"
NOTE_TAG = "{" + TEI + "}note"
SENTENCE_TAG = "{" + TEI + "}s"
P_TAG = "{" + TEI + "}p"
WORD_TAG = "{" + TEI + "}w"
PUNCTUATION_TAG = "{" + TEI + "}pc"

# Characters that are 100% not in the xml files ()
CHARACTERS_TO_REMOVE: set[str] = {'@', '#', '$', '^', '&', '*', '<', '>', '­'}


def get_associated_pdf(xml_path: str) -> str:
    xml_tree = ET.parse(xml_path)
    xml_root: ET.Element = xml_tree.getroot()

    pdf_title_element: ET.Element = xml_root.find(".//tei:title[@type='pdf']", {"tei": TEI})

    pdf_path: str = pdf_title_element.text.replace('../yu1Parl-source', PATH_TO_PDF_FILES)

    return pdf_path


def get_elements_by_tags(root: ET.Element, wanted_tags: set[str]) -> list[ET.Element]:
    elements = []
    for child in root:
        if child.tag in wanted_tags:
            elements.append(child)
        else:
            elements.extend(get_elements_by_tags(child, wanted_tags))

    return elements


def get_chars_from_pdf(pdf_path: str) -> list[dict]:
    pdf_chars: list[dict] = []

    # Collect all characters from the PDF into a list
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, pdf_page in enumerate(pdf.pages):
            chars_on_page: list[dict] = pdf_page.chars

            if not chars_on_page:
                continue

            pdf_chars.extend(chars_on_page)

    return pdf_chars


def get_converter_function(xml_sentences: list[ET.Element]) -> Callable:
    # Count the number of sentences in serbian
    number_of_sentences_in_serbian: int = 0
    for sentence in xml_sentences:
        if "{" + NAMESPACE + "}lang" in sentence.attrib and sentence.attrib["{" + NAMESPACE + "}lang"] == "sr":
            number_of_sentences_in_serbian += 1

    # If most of the sentences is in serbian return to_cyrillic function else to_latin
    if number_of_sentences_in_serbian / len(xml_sentences) > 0.5:
        return to_cyrillic
    else:
        return to_latin


def save_xml_tree(xml_tree: ET.ElementTree, output_file: str) -> None:
    ET.register_namespace('', TEI)
    xml_tree.write(output_file, encoding='utf-8')


def get_locations_to_remove(alignment: str, seq: str) -> set[int]:
    locations_to_remove: list[tuple[int, int]] = []
    start = None  # To mark the start of a sequence of '-'
    noise_count = 0  # To count the '|' characters within a sequence

    for i, char in enumerate(alignment):
        if char == '-':
            if start is None:
                start = i  # Mark the start of a sequence
                noise_count = 0  # Reset noise count
        else:
            if start is not None:
                if all([c != '-' for c in alignment[i:i + 10]]):
                    locations_to_remove.append((start, i - noise_count))  # End of a sequence
                    start = None
                else:
                    noise_count += 1  # Increment noise count

    if start is not None:
        locations_to_remove.append((start, len(alignment)))

    l_t_r_set = set()
    for s, e in locations_to_remove:
        # print(seq[s:e])
        for i in range(s, e):
            l_t_r_set.add(i)

    return l_t_r_set


def additional_align(xml_element: ET.Element, pdf_chars: list[dict], converter_function: Callable) -> list[dict]:
    target: str = get_text_from_element(xml_element)
    sequence: str = "".join([char["text"] for char in pdf_chars])

    target = re.sub(r'\s+|\t|\n|\r', '', target)
    sequence = re.sub(r'\s+|\t|\n|\r', '', sequence)

    result = edlib.align(
        converter_function(target, "sr"),
        converter_function(sequence, "sr"),
        task="path", mode="NW",
    )

    alignment = edlib.getNiceAlignment(result, target, sequence)

    indices_to_remove = get_locations_to_remove(alignment["matched_aligned"], sequence)

    # Return a new list without those indices
    filtered = [char for i, char in enumerate(pdf_chars) if i not in indices_to_remove]

    return filtered


def get_text_from_element(element: ET.Element) -> str:
    if element.tag in {WORD_TAG, PUNCTUATION_TAG, NOTE_TAG}:
        return element.text

    return "".join([get_text_from_element(child) for child in element])


def remove_unwanted_chars(pdf_chars: list[dict], unwanted_chars: set[str]) -> list[dict]:
    return [char for char in pdf_chars if char['text'] not in unwanted_chars and not char['text'].isspace()]


def is_duplicate_note_element(element: ET.Element) -> bool:
    return element.tag == NOTE_TAG and element.attrib.get("subtype") == "latin"


def visualize_xml(xml_root: ET.Element, xml_path: str, pdf_path: str) -> None:
    xml_elements = get_elements_by_tags(xml_root, {WORD_TAG, PUNCTUATION_TAG})

    base_name = os.path.basename(xml_path).replace('.tei.xml', '')

    # Make a list of images of each page in the PDF
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            images.append(page.to_image(resolution=150))

    # For each word in the XML draw a rectangle around it on the PDF
    for xml_element in xml_elements:

        # Skip elements that are considered as noise
        if 'fromPage' not in xml_element.attrib or 'toPage' not in xml_element.attrib or 'x0' not in xml_element.attrib or 'y0' not in xml_element.attrib:
            continue

        try:
            fromPage = int(xml_element.attrib['fromPage'])
            toPage = int(xml_element.attrib['toPage'])

            x_coords = [float(xml_element.attrib[key]) for key in xml_element.attrib if key.startswith('x')]
            y_coords = [float(xml_element.attrib[key]) for key in xml_element.attrib if key.startswith('y')]

            i = 0
            for x0, y0, x1, y1 in zip(x_coords[::2], y_coords[::2], x_coords[1::2], y_coords[1::2]):
                page_num = fromPage if i == 0 else toPage

                images[page_num].draw_rect((x0, y0, x1, y1), stroke_width=1)
                i += 1
        except:
            print(
                f"visualize_xml(): COORD ERROR with word: '{xml_element.text}', {xml_element.attrib.get('{' + TEI + '}id')}")

    # Save the images
    if not os.path.exists(os.path.join(VISUALIZATION_FILE, base_name)):
        os.makedirs(os.path.join(VISUALIZATION_FILE, base_name))
    for i, image in enumerate(images):
        image.save(os.path.join(VISUALIZATION_FILE, base_name, f"{base_name}_{i}.png"), )


# Adds coordinates to the xml element
def add_metadata_to_word_element(xml_element: ET.Element, pdf_chars: list[dict]) -> None:
    coord_counter: int = 0

    for i, char in enumerate(pdf_chars):
        if i == 0:
            xml_element.set(f'x{coord_counter}', str(round(char['x0'], 2)))
            xml_element.set(f'y{coord_counter}', str(round(char['top'], 2)))
            xml_element.set('fromPage', str(char['page_number'] - 1))
            xml_element.set('isBroken', 'false')
            coord_counter += 1

        if 0 <= i < len(pdf_chars) - 1 and \
                i + 1 < len(pdf_chars) and \
                abs(int(char['bottom']) - int(pdf_chars[i + 1]['bottom'])) >= 4:
            # end of previous part of the word
            xml_element.set(f'x{coord_counter}', str(round(char['x1'], 2)))
            xml_element.set(f'y{coord_counter}', str(round(char['bottom'], 2)))
            coord_counter += 1
            # start of new part of the word
            xml_element.set(f'x{coord_counter}', str(round(pdf_chars[i + 1]['x0'], 2)))
            xml_element.set(f'y{coord_counter}', str(round(pdf_chars[i + 1]['top'], 2)))
            coord_counter += 1

            xml_element.set('isBroken', 'true')

        if i == len(pdf_chars) - 1:
            xml_element.set(f'x{coord_counter}', str(round(char['x1'], 2)))
            xml_element.set(f'y{coord_counter}', str(round(char['bottom'], 2)))
            xml_element.set('toPage', str(char['page_number'] - 1))


def parse_sentence(pdf_chars: list[dict], xml_segment: ET.Element, converter_function: Callable) -> None:
    xml_words: list[ET.Element] = get_elements_by_tags(xml_segment, {WORD_TAG, PUNCTUATION_TAG})
    sequence: str = "".join([char['text'] for char in pdf_chars])

    # Define search area window
    search_from: int = 0

    for xml_word in xml_words:

        target: str = re.sub(r'\s+|\t|\n|\r', '', get_text_from_element(xml_word))
        if len(target) == 0:
            continue

        similarity_curr: float = 0
        similarity_prev: float = -1
        BUFFER: int = len(target) // 2

        while similarity_prev < similarity_curr < 1.0:
            # adjust searching area while searching for the target sentence
            search_area_start: int = search_from
            search_area_end: int = min(search_area_start + len(target) + BUFFER, len(pdf_chars))
            search_area: str = sequence[search_area_start:search_area_end]

            # Perform alignment
            result = edlib.align(
                converter_function(target, "sr"),
                converter_function(search_area, "sr"),
                task="path", mode="HW",
            )

            # Update similarity values
            similarity_prev = similarity_curr
            similarity_curr = 1 - result['editDistance'] / len(target)

            BUFFER += 1

        # Skip current element if there is no match
        if result['locations'][0][0] is None:
            continue

        best_match_start: int = search_from + result['locations'][0][0]
        best_match_end: int = search_from + result['locations'][0][-1] + 1

        search_from = best_match_end

        print("Simil:", similarity_curr)
        wr_id = xml_word.attrib["{" + NAMESPACE + "}id"]
        print("Wr_id:", wr_id)
        print("Word :", target)
        print("Match:", "".join([c["text"] for c in pdf_chars[best_match_start:best_match_end]]))
        print()

        if similarity_curr < 0.7:
            continue

        add_metadata_to_word_element(xml_word, pdf_chars[best_match_start: best_match_end])


"""
def parse_segment(pdf_chars1: list[dict], xml_segment: ET.Element, converter_function: Callable) -> None:
    xml_sentences: list[ET.Element] = get_elements_by_tags(xml_segment, {SENTENCE_TAG, NOTE_TAG})
    sequence: str = "".join([c['text'] for c in pdf_chars1])

    # Define search area window
    search_area_start: int = 0
    search_area_end: int = 0

    # For each sentence, try to find the optimal alignment in the segment
    for xml_sentence in xml_sentences:
        target: str = re.sub(r'\s+|\t|\n|\r', '', get_text_from_element(xml_sentence))
        if len(target) == 0:
            continue

        max_attempts: int = 3
        attempt: int = 0
        success: bool = False  # flag to mark if we got a good enough similarity
        result = None
        similarity_curr: float = 0
        resync: bool = False

        # Retry the alignment process until similarity >= 0.75 or we run out of attempts
        while attempt < max_attempts and not success:
            attempt += 1
            similarity_curr = 0
            similarity_prev: float = -1
            BUFFER: int = min(len(target) // 2, 5)

            # Inner loop: gradually expand the search area until improvement stops
            while round(similarity_prev, 2) < round(similarity_curr, 2) < 1.0:
                search_area_start = search_area_end
                search_area_end = search_area_end + len(target) + BUFFER
                if resync:
                    search_area_start = 0
                    search_area_end = len(sequence)
                    resync = False
                search_area: str = sequence[search_area_start: search_area_end]

                result = edlib.align(
                    converter_function(target, "sr"),
                    converter_function(search_area, "sr"),
                    task="path", mode="HW",
                )

                similarity_prev = similarity_curr
                similarity_curr = 1 - result['editDistance'] / len(target)
                BUFFER += 1

            # If no valid location is found, break out
            if result['locations'][0][0] is None:
                break

            # Determine the actual matching area
            search_area_start = search_area_start + result['locations'][0][0]
            search_area_end = search_area_start + result['locations'][0][-1] + 1

            if similarity_curr < 0.5:
                # Not similar enough—try again by reiterating the alignment process.
                # (You might consider additional logic here such as logging or modifying parameters.)
                resync = True
                continue  # This will repeat the while loop for this sentence
            else:
                # Found a sufficiently similar alignment
                success = True
                break

        print("Simil:", similarity_curr)
        s_id = xml_sentence.attrib[
            "{" + NAMESPACE + "}id"] if "{" + NAMESPACE + "}id" in xml_sentence.attrib else "note"
        print("Sn_id:", s_id)
        print("Sente:", target)
        print("Match:", "".join([c["text"] for c in pdf_chars1[search_area_start:search_area_end]]))
        print()

        # If after max attempts the similarity is still too low or no valid alignment was found, skip this sentence.
        if not success:
            continue

        # Skip processing if the element is a note
        if xml_sentence.tag == NOTE_TAG:
            continue

        # Process the sentence alignment match
        # parse_sentence(pdf_chars1[search_area_start:search_area_end], xml_sentence, converter_function)
"""


def parse_segment(pdf_chars: list[dict], xml_segment: ET.Element, converter_function: Callable) -> None:
    xml_senteces: list[ET.Element] = get_elements_by_tags(xml_segment, {SENTENCE_TAG, NOTE_TAG})
    sequence: str = "".join([c['text'] for c in pdf_chars])

    # Define search area window
    search_from: int = 0

    # For each sentence find optimal alignment in the segment
    for xml_sentence in xml_senteces:

        target: str = re.sub(r'\s+|\t|\n|\r', '', get_text_from_element(xml_sentence))
        if len(target) == 0:
            continue

        similarity_curr: float = 0
        similarity_prev: float = -1
        BUFFER: int = len(target) // 2

        while similarity_prev < similarity_curr < 1.0:
            # adjust searching area while searching for the target sentence
            search_area_start: int = search_from
            search_area_end: int = min(search_area_start + len(target) + BUFFER, len(pdf_chars))
            search_area: str = sequence[search_area_start:search_area_end]

            # Perform alignment
            result = edlib.align(
                converter_function(target, "sr"),
                converter_function(search_area, "sr"),
                task="path", mode="HW",
            )

            # Update similarity values
            similarity_prev = similarity_curr
            similarity_curr = 1 - result['editDistance'] / len(target)

            BUFFER += 1

        # Skip current element if there is no match
        if result['locations'][0][0] is None:
            continue

        best_match_start: int = search_from + result['locations'][0][0]
        best_match_end: int = search_from + result['locations'][0][-1] + 1

        search_from = best_match_end

        print("Simil:", similarity_curr)
        s_id = xml_sentence.attrib[
            "{" + NAMESPACE + "}id"] if "{" + NAMESPACE + "}id" in xml_sentence.attrib else "note"
        print("Sn_id:", s_id)
        print("Targt:", target)
        print("Match:", "".join([c["text"] for c in pdf_chars[best_match_start:best_match_end]]))
        print()

        if xml_sentence.tag == NOTE_TAG:
            continue

        # if similarity_curr < 0.7:
        #     continue

        cleared_pdf_chars: list[dict] = additional_align(
            xml_sentence,
            pdf_chars[max(best_match_start - 22, 0):min(best_match_end + 22, len(pdf_chars))],
            converter_function
        )

        parse_sentence(cleared_pdf_chars, xml_sentence, converter_function)


def parse_record(xml_path: str, pdf_path: str) -> None:
    xml_tree: ET.ElementTree = ET.parse(xml_path)
    xml_root: ET.Element = xml_tree.getroot()

    # 1. Get segments from XML
    print("parse_record(): Getting segments from XML")
    xml_segments: list[ET.Element] = get_elements_by_tags(xml_root, {SEGMENT_TAG, NOTE_TAG})
    # 1.1. Remove duplicate note elements
    xml_segments = [element for element in xml_segments if not is_duplicate_note_element(element)]

    # 2. Get all characters from the PDF
    print("parse_record(): Getting characters from PDF")
    pdf_chars: list[dict] = get_chars_from_pdf(pdf_path)
    # 2.1. Remove unwanted characters
    pdf_chars = remove_unwanted_chars(pdf_chars, CHARACTERS_TO_REMOVE)

    # 3. Get the correct function to convert text (to_latin or to_cyrillic) based on lang in sentence elements
    xml_sentences: list[ET.Element] = xml_root.findall(".//tei:s", {"tei": TEI})
    converter_function: Callable = get_converter_function(xml_sentences)

    # 4. Aligning segments and skipping those that are most likely a table
    print("parse_record(): Parsing segments")
    sequence: str = "".join([char['text'] for char in pdf_chars])

    for i, xml_segment in enumerate(xml_segments):

        target: str = re.sub(r'\s+|\t|\n|\r', '', get_text_from_element(xml_segment))
        if len(target) == 0:
            continue

        # Perform alignment
        result = edlib.align(
            converter_function(target, "sr"),
            converter_function(sequence, "sr"),
            task="path", mode="HW",
        )

        similarity: float = 1 - result["editDistance"] / len(target)
        # skip note elements and segments that are probably table
        if similarity < 0.7:
            print(xml_segment.attrib[
                      "{" + NAMESPACE + "}id"] if "{" + NAMESPACE + "}id" in xml_segment.attrib else "note")
            continue

        segment_start: int = result["locations"][0][0]
        segment_end: int = result["locations"][0][-1] + 1

        if xml_segment.tag == NOTE_TAG:
            continue

        BUFFER: int = max(len(target) // 2, 30)

        cleared_pdf_chars: list[dict] = additional_align(
            xml_segment,
            pdf_chars[segment_start - 50: segment_end + 50],
            converter_function
        )

        # if "{" + NAMESPACE + "}id" in xml_segment.attrib and xml_segment.attrib[
        #     "{" + NAMESPACE + "}id"] == "yu1Parl_1919-05-23_PP_24_seg3":
        print("Simil:", similarity)
        print("Sg_id:",
              xml_segment.attrib["{" + NAMESPACE + "}id"] if "{" + NAMESPACE + "}id" in xml_segment.attrib else "note")
        print("Targt:", target)
        # print("Match1:", "".join([c["text"] for c in pdf_chars[segment_start - 30: segment_end + 30]]))
        print("Match2:", "".join([c["text"] for c in cleared_pdf_chars]))
        print()

        parse_segment(pdf_chars[segment_start - 50: segment_end + 50], xml_segment, converter_function)

    # Save the updated XML content
    if not os.path.exists(OUTPUT_FILE):
        os.makedirs(OUTPUT_FILE)
    save_xml_tree(xml_tree, os.path.join(OUTPUT_FILE, os.path.basename(xml_path)))

    if VISUALIZE_COORDINATES_FROM_XML:
        print("parse_record(): Visualizing coordinates")
        visualize_xml(xml_root, xml_path, pdf_path)


def main() -> None:
    files_converted = 0

    # xml_files = sorted(os.listdir(PATH_TO_XML_FILES),
    #                   key=lambda f: os.path.getsize(os.path.join(PATH_TO_XML_FILES, f)))

    xml_files = sorted(os.listdir(PATH_TO_XML_FILES))
    for i, xml_file in enumerate(xml_files):

        xml_path = os.path.join(PATH_TO_XML_FILES, xml_file)

        if not xml_path.endswith('.xml'):
            continue

        if MAX_FILES - files_converted == 0:
            break

        if files_converted < SKIP_FILES_TO:
            files_converted += 1
            continue

        print(f"Processing file: {xml_file}")
        pdf_path: str = get_associated_pdf(xml_path)

        time_start = time.time()
        parse_record(xml_path, pdf_path)
        time_end = time.time()

        print(f"Time taken: {time_end - time_start} seconds")

        files_converted += 1

        print(f"Progress: {files_converted}/{len(xml_files)}\n\n")


if __name__ == '__main__':
    main()
