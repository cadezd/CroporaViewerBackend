import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from platform import system

import docx
from cyrtranslit import to_latin, to_cyrillic

import edlib
import pdfplumber

PATH_TO_XML_FILES = "/home/davidlocal/raw-data/yu1Parl.TEI.ana"
PATH_TO_PDF_FILES = "/home/davidlocal/raw-data/yu1Parl-source"
PATH_TO_WORD_FILES = "/home/davidlocal/raw-data/yu1Parl-source"
OUTPUT_FILE = "/home/davidlocal/raw-data/yuparl-xml-enriched"

# Set to True if you want to visualize the coordinates on the PDF and save the images into a folder
VISUALIZE_COORDINATES_FROM_XML = True
VISUALIZATION_FILE = "/home/davidlocal/raw-data/yuparl-visualizations"

SKIP_FILES_TO = 0  # set to 0 if you want to convert all files
MAX_FILES = 15  # set to -1 if you want to convert all files

# If you want to see alignment for each word in the sentence set this to True
# Target -> word from the xml; Best match -> word from the pdf; Similarity -> similarity between the two words
PRINT_ALIGNMENT = True

# Namespace
TEI = "http://www.tei-c.org/ns/1.0"

# Tags in the XML files
SEGMENT_TAG = "{" + TEI + "}seg"
NOTE_TAG = "{" + TEI + "}note"
SENTENCE_TAG = "{" + TEI + "}s"
P_TAG = "{" + TEI + "}p"
WORD_TAG = "{" + TEI + "}w"
PUNCTUATION_TAG = "{" + TEI + "}pc"

# Characters that are 100% not in the xml files ()
CHARACTERS_TO_REMOVE: set[str] = {'@', '#', '$', '^', '&', '*', '<', '>', '­',}

ADDITIONAL_EQUALITIES = [
    ('2', 'М'),
    ('3', 'и'),
]


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


def get_associated_pdf(xml_path: str) -> str:
    print(xml_path)
    xml_tree = ET.parse(xml_path)
    xml_root: ET.Element = xml_tree.getroot()

    pdf_title_element: ET.Element = xml_root.find(".//tei:title[@type='pdf']", {"tei": TEI})

    pdf_path: str = pdf_title_element.text.replace('../yu1Parl-source', PATH_TO_PDF_FILES)

    return pdf_path


def get_associated_word(xml_path: str) -> str:
    print(xml_path)
    xml_tree = ET.parse(xml_path)
    xml_root: ET.Element = xml_tree.getroot()

    pdf_title_element: ET.Element = xml_root.find(".//tei:title[@type='docx']", {"tei": TEI})

    word_path: str = pdf_title_element.text.replace('../yu1Parl-source', PATH_TO_WORD_FILES)

    return word_path


def get_elements_by_tags(root: ET.Element, wanted_tags: set[str]) -> list[ET.Element]:
    elements = []
    for child in root:
        if child.tag in wanted_tags:
            elements.append(child)
        elements.extend(get_elements_by_tags(child, wanted_tags))
    return elements


def get_text_from_element(element: ET.Element) -> str:
    if element.tag in {WORD_TAG, PUNCTUATION_TAG, NOTE_TAG}:
        return element.text

    return "".join([child.text for child in element])


def get_position_of_target_in_sequence(target: str, sequence: str, last_occurrence: bool = False) -> tuple[int, int]:
    results: dict = edlib.align(
        to_cyrillic(target),
        to_cyrillic(sequence),
        task="path",
        mode="HW",
    )

    occurrence = 0 if not last_occurrence else -1

    return results['locations'][occurrence]


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


def save_xml_tree(xml_tree: ET.ElementTree, output_file: str) -> None:
    ET.register_namespace('', TEI)
    xml_tree.write(output_file, encoding='utf-8')


# Filters out chars that are not part of the session content (before the session_start_str occours)
def get_session_content(pdf_chars: list[dict], session_start_str: str | None) -> \
        list[dict]:
    # session_start_str is string below which we start extracting text (start of the session content)

    # If the session start or end notes are not provided, return the original list of characters
    if session_start_str is None:
        return pdf_chars

    sequence: str = "".join([char['text'] for char in pdf_chars])
    sequence = re.sub(r'\s+|\t|\n|\r', '', sequence)

    session_start_idx: int = get_position_of_target_in_sequence(session_start_str, sequence)[0]

    # Necessary parameters for filtering out the session content
    first_page: int = pdf_chars[session_start_idx]['page_number']
    first_page_session_start_y: float = pdf_chars[session_start_idx]['top'] - 10

    # Filter out the session content
    session_content: list[dict] = []
    for char in pdf_chars:
        if (char["page_number"] <= first_page and
                (char['page_number'] != first_page or char['top'] < first_page_session_start_y)
        ):
            continue

        session_content.append(char)

    return session_content


# Remove unwanted characters and whitespaces (improves the alignment and search)
def remove_unwanted_chars(pdf_chars: list[dict], unwanted_chars: set[str]) -> list[dict]:
    return [char for char in pdf_chars if char['text'] not in unwanted_chars and not char['text'].isspace()]


def get_locations_to_remove(alignment: str) -> set[int]:
    """
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
                if all([c != '-' for c in alignment[i + 1:i + 1 + 6]]):
                    locations_to_remove.append((start, i - noise_count))  # End of a sequence
                    start = None
                else:
                    noise_count += 1  # Increment noise count

    if start is not None:
        locations_to_remove.append((start, len(alignment)))

    l_t_r_set = set()
    for s, e in locations_to_remove:
        l_t_r_set.add(range(s, e))

    return l_t_r_set
    """


    locations_to_remove = set()
    for i, chr in enumerate(alignment):
        if chr == "-":
            locations_to_remove.add(i)

    return locations_to_remove



def get_text_from_word_doc(filename: str) -> str:
    doc = docx.Document(filename)
    all_chars = []
    for para in doc.paragraphs:
        for run in para.runs:
            all_chars.extend(list(run.text))

    return ''.join(all_chars)


def align_pdf_with_xml(xml_sentences: list[ET.Element], pdf_chars: list[dict]) -> list[dict]:
    target = "".join([to_cyrillic(get_text_from_element(element)) for element in xml_sentences])
    target = re.sub(r'\s+|\t|\n|\r', '', target)
    sequence: str = "".join([re.sub(r'\s+|\t|\n|\r', '', to_cyrillic(char["text"])) for char in pdf_chars])

    result = edlib.align(
        target,
        sequence,
        task="path", mode="NW",
    )

    alignment = edlib.getNiceAlignment(result, target, sequence)

    indices_to_remove = get_locations_to_remove(alignment["matched_aligned"])

    # Return a new list without those indices
    filtered =  [char for i, char in enumerate(pdf_chars) if i not in indices_to_remove]

    return filtered


# Adds coordinates to the xml element
def add_metadata(xml_element: ET.Element, pdf_chars: list[dict]) -> None:
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


# Extracts coordinates for each word in a sentence
def parse_words(pdf_chars: list[dict], xml_sentence: ET.Element):
    elements_in_sentence: list[ET.Element] = get_elements_by_tags(xml_sentence, {WORD_TAG, PUNCTUATION_TAG})

    sequence: str = "".join([re.sub(r'\s+|\t|\n|\r', '', char["text"]) for char in pdf_chars])

    target_sequnece: str = "".join([get_text_from_element(xml_element) for xml_element in elements_in_sentence])

    if PRINT_ALIGNMENT:
        print("\nSentenceId", xml_sentence.attrib["{http://www.w3.org/XML/1998/namespace}id"])
        print("Sentence:", sequence)
        print("Target:", target_sequnece)

    best_match_end: int = 0
    search_from: int = 0
    result: dict = None
    while elements_in_sentence:
        xml_element: ET.Element = elements_in_sentence.pop(0)
        target: str = re.sub(r'\s+|\t|\n|\r', '', get_text_from_element(xml_element))

        similarity_curr: float = 0
        similarity_prev: float = -1
        BUFFER: int = 2

        while round(similarity_prev, 2) < round(similarity_curr, 2) < 1.0:
            search_area_start: int = search_from
            search_area_end: int = search_area_start + len(target) + BUFFER
            search_area: str = sequence[search_area_start:search_area_end]

            result = edlib.align(
                to_cyrillic(target),
                to_cyrillic(search_area),
                task="path", mode='HW',
            )

            similarity_prev = similarity_curr
            similarity_curr = 1 - result['editDistance'] / max(len(target), len(search_area))

            BUFFER += 1

        if result['locations'][0][0] is None:
            continue

        best_match_start: int = search_from + result['locations'][0][0]
        best_match_end: int = search_from + result['locations'][0][-1]

        # Move the search area forward in case the target is of length 1
        # search_from = best_match_end if best_match_end > search_from or \
        #                                (elements_in_sentence and len(
        #                                    elements_in_sentence[0].text) > 1) else best_match_end + 1

        search_from = best_match_end + 1

        if PRINT_ALIGNMENT:
            print(
                f"Target: {target: <25} Best match: {sequence[best_match_start:best_match_end + 1]: <25} Similarity: {similarity_curr:.2f}")

        # Add coordinates to xml element
        add_metadata(xml_element, pdf_chars[best_match_start:best_match_end + 1])


def parse_record(xml_path: str, pdf_path: str, word_path: str) -> None:
    xml_tree = ET.parse(xml_path)
    xml_root: ET.Element = xml_tree.getroot()

    # 1. Get all elements from the XML that are part of the session content
    print("parse_record(): Getting session content from XML")
    session_xml_content: list[ET.Element] = get_elements_by_tags(xml_root, {SENTENCE_TAG, NOTE_TAG, P_TAG})
    session_xml_content = [element for element in session_xml_content if element or element.text]

    # 2. Filter out notes with type speaker and subtype="latin"
    session_xml_content = [element for element in session_xml_content if
                           not (element.tag == NOTE_TAG and element.attrib.get("subtype") == "latin")]

    # 3. Get all characters from the PDF and perform filtering
    print("parse_record(): Getting characters from PDF")
    pdf_chars: list[dict] = get_chars_from_pdf(pdf_path)

    # 4. Get note that indicate the start of the session
    print("parse_record(): Filtering unnecessary characters")
    notes: list[ET.Element] = get_elements_by_tags(xml_root, {NOTE_TAG})
    session_start_note: ET.Element = notes[0]

    # 5. Keep only characters that are part of the session content (after start note)
    session_pdf_content = get_session_content(
        pdf_chars,
       # session_start_note.text if session_start_note.text else None,
        None
    )

    # 6. Remove unwanted characters
    session_pdf_content = remove_unwanted_chars(session_pdf_content, CHARACTERS_TO_REMOVE)

    # 7. Align the XML content with the PDF content (to remove any text from the PDF that is not in the XML)
    session_pdf_content = align_pdf_with_xml(session_xml_content, session_pdf_content)


    # Add the coordinates to the XML content
    print("parse_record(): Adding metadata to XML")

    best_match_end: int = 0
    search_area_start: int = 0
    sequence: str = "".join([char['text'] for char in session_pdf_content])
    while session_xml_content:

        # Sentence or note element
        xml_element: ET.Element = session_xml_content.pop(0)
        target: str = re.sub(r'\s+|\t|\n|\r', '', get_text_from_element(xml_element))

        if len(target) == 0:
            continue

        result: dict = None
        similarity_curr: float = 0
        similarity_prev: float = -1
        BUFFER: int = 5

        while round(similarity_prev, 2) < round(similarity_curr, 2) < 1.0:
            # adjust searching area while searching for the target sentence
            search_area_start: int = best_match_end
            search_area_end: int = search_area_start + len(target) + BUFFER
            search_area: str = sequence[search_area_start:search_area_end]

            # Perform alignment
            result = edlib.align(
                to_cyrillic(target),
                to_cyrillic(search_area),
                task="path", mode="HW",
            )

            # Update similarity values
            similarity_prev = similarity_curr
            similarity_curr = 1 - result['editDistance'] / max(len(target), len(search_area))

            BUFFER += 1

        # Skip current element if there is no match
        if result['locations'][0][0] is None:
            continue

        best_match_start: int = search_area_start + result['locations'][0][0]
        best_match_end: int = search_area_start + result['locations'][0][-1]

        # Skip elements that are not sentence
        if xml_element.tag == NOTE_TAG:
            continue

        parse_words(session_pdf_content[best_match_start:best_match_end + 1], xml_element)

    # Save the updated XML content
    if not os.path.exists(OUTPUT_FILE):
        os.makedirs(OUTPUT_FILE)
    save_xml_tree(xml_tree, os.path.join(OUTPUT_FILE, os.path.basename(xml_path)))

    if VISUALIZE_COORDINATES_FROM_XML:
        print("parse_record(): Visualizing coordinates")
        visualize_xml(xml_root, xml_path, pdf_path)


def main() -> None:
    files_converted = 0

    xml_files = sorted(os.listdir(PATH_TO_XML_FILES), key=lambda f: os.path.getsize(os.path.join(PATH_TO_XML_FILES, f)))[15:]
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
        word_path: str = get_associated_word(xml_path)

        time_start = time.time()

        parse_record(xml_path, pdf_path, word_path)

        time_end = time.time()

        print(f"Time taken: {time_end - time_start} seconds")

        files_converted += 1

        print(f"Progress: {files_converted}/{len(xml_files)}\n\n")


if __name__ == '__main__':
    main()
