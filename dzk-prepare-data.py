import os
import shutil

import fitz

SOURCE_DIRECTORY = "C:\\Users\\david\\fakulteta\\diplomska\\all_pdf\\"
DESTINATION_DIRECTORY = "C:\\Users\\david\\OneDrive\\Desktop\\"


def copy_pdfs(SOURCE_DIRECTORY, DESTINATION_DIRECTORY):
    for file in os.listdir(SOURCE_DIRECTORY):
        if file.endswith(".pdf"):
            print(f'copy_pdfs(): copying file {file:.<70}', end='')
            file_path = os.path.join(SOURCE_DIRECTORY, file)
            shutil.copy(file_path, DESTINATION_DIRECTORY)
            print('DONE')


def create_thumbnails(SOURCE_DIRECTORY, DESTINATION_DIRECTORY):
    for file in os.listdir(SOURCE_DIRECTORY):
        if file.endswith(".pdf"):
            file_path = os.path.join(SOURCE_DIRECTORY, file)
            print(f'create_thumbnails(): creating thumbnail for {file:.<70}', end='')

            # get first page of pdf
            pdf_document = fitz.open(file_path)
            first_page = pdf_document[0]

            # generate thumbnail
            image = first_page.get_pixmap()
            image.save(os.path.join(DESTINATION_DIRECTORY, f"{file.split('.')[0]}.png"))

            print('DONE')


# rename all files in directory to DZK_YYYY-MM-DD_VOLUME_NUMBER format
def rename_files_in_dir(path):
    for file in os.listdir(path):
        print(f'rename_files_in_dir(): renaming file {file:.<70}', end='')

        # date
        date = file.split("-")[1]
        year = date[:4]
        month = date[4:6]
        day = date[6:]

        # volume
        volume = file.split("-")[2]
        # and number
        number = file.split("-")[3]

        # rename file
        new_file_name = f"DZK_{year}-{month}-{day}_{volume}_{number}"
        os.rename(os.path.join(path, file),
                  os.path.join(path, new_file_name))

        print(f'in {new_file_name:.<30}')


if __name__ == '__main__':
    DESTINATION_DIRECTORY = os.path.join(DESTINATION_DIRECTORY, "DZK")
    DESTINATION_DIRECTORY_PDFS = os.path.join(DESTINATION_DIRECTORY, "Kranjska-pdf")
    DESTINATION_DIRECTORY_THUMBNAILS = os.path.join(DESTINATION_DIRECTORY, "thumbnails")
    os.makedirs(DESTINATION_DIRECTORY, exist_ok=True)
    os.makedirs(DESTINATION_DIRECTORY_PDFS, exist_ok=True)
    os.makedirs(DESTINATION_DIRECTORY_THUMBNAILS, exist_ok=True)
    copy_pdfs(SOURCE_DIRECTORY, DESTINATION_DIRECTORY_PDFS)
    rename_files_in_dir(DESTINATION_DIRECTORY_PDFS)
    create_thumbnails(DESTINATION_DIRECTORY_PDFS, DESTINATION_DIRECTORY_THUMBNAILS)
    print('\n\n ALL DONE')
