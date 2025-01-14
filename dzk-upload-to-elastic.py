import os

from elasticsearch import Elasticsearch, helpers

# Path to the directory containing the attendees, places, meeting, sentences, and words JSONL files
PATH_TO_JSONL_FILES = "D:\\diplomska-data\\second-parsing\\"

DELETE_INDEX_IF_EXISTS = False

SKIP_FILES_TO = 0  # set to 0 if you want to convert all files
MAX_FILES = -1  # set to -1 if you want to convert all files

# initialize the Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}], max_retries=20, request_timeout=180)

MEETINGS_INDEX_NAME = "meetings-index"
SENTENCES_INDEX_NAME = "sentences-index"
WORDS_INDEX_NAME = "words-index"
PLACES_INDEX_NAME = "places-index"
ATTENDEES_INDEX_NAME = "attendees-index"

# Settings for the Elasticsearch indices (mappings, analyzers, etc.)
MEETINGS_INDEX_SETTINGS = {
    "index.mapping.total_fields.limit": 10000000,
    "index.mapping.nested_objects.limit": 10000000,
    "index.max_inner_result_window": 10000,
    "index.max_result_window": 10000,
    "index.number_of_replicas": 2,
    "index.refresh_interval": "30s",
}
MEETINGS_INDEX_MAPPING = {
    "properties": {
        "agendas": {
            "properties": {
                "items": {
                    "properties": {
                        "n": {
                            "type": "long"
                        },
                        "text": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        }
                    }
                },
                "lang": {
                    "type": "keyword"
                }
            }
        },
        "corpus": {
            "type": "keyword"
        },
        "date": {
            "type": "date",
            "format": "dd.MM.yyyy"
        },
        "id": {
            "type": "keyword",
        },
        "notes": {
            "properties": {
                "page": {
                    "type": "keyword"
                },
                "segment_id": {
                    "type": "keyword",

                },
                "speaker": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "text": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "type": {
                    "type": "keyword"
                }
            }
        },
        "sentences": {
            "properties": {
                "id": {
                    "type": "keyword",
                },
                "original_language": {
                    "type": "keyword"
                },
                "segment_id": {
                    "type": "keyword",
                },
                "segment_page": {
                    "type": "long"
                },
                "speaker": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "translations": {
                    "type": "nested",
                    "properties": {
                        "lang": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "original": {
                            "type": "long"
                        },
                        "speaker": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "text": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "words": {
                            "properties": {
                                "id": {
                                    "type": "keyword",
                                },
                                "lemma": {
                                    "type": "text",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword",
                                            "ignore_above": 256
                                        }
                                    }
                                },
                                "text": {
                                    "type": "text",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword",
                                            "ignore_above": 256
                                        }
                                    }
                                },
                                "type": {
                                    "type": "keyword"
                                },
                                "join": {
                                    "type": "keyword"
                                },
                                "propn": {
                                    "type": "long"
                                },
                            }
                        }
                    }
                }
            }
        },
        "titles": {
            "properties": {
                "lang": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "title": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                }
            }
        }
    }
}

SENTENCES_INDEX_SETTINGS = {
    "index.max_inner_result_window": 50000,
    "index.max_result_window": 50000,
    "index.number_of_replicas": 2,
    "index.refresh_interval": "30s",
    "analysis": {
        "filter": {
            "custom_ascii_folding": {
                "type": "asciifolding",
                "preserve_original": "false"
            }
        },
        "analyzer": {
            "custom_text_analyzer": {
                "filter": [
                    "custom_ascii_folding",
                    "lowercase"
                ],
                "type": "custom",
                "tokenizer": "standard"
            }
        }
    }
}
SENTENCES_INDEX_MAPPING = {
    "properties": {
        "coordinates": {
            "properties": {
                "page": {
                    "type": "long"
                },
                "x0": {
                    "type": "float"
                },
                "x1": {
                    "type": "float"
                },
                "y0": {
                    "type": "float"
                },
                "y1": {
                    "type": "float"
                }
            }
        },
        "meeting_id": {
            "type": "keyword",
            "fields": {
                "sort": {
                    "type": "icu_collation_keyword",
                    "index": False,
                    "numeric": True
                }
            }
        },
        "segment_id": {
            "type": "keyword",
            "fields": {
                "sort": {
                    "type": "icu_collation_keyword",
                    "index": False,
                    "numeric": True
                }
            }
        },
        "sentence_id": {
            "type": "keyword",
            "fields": {
                "sort": {
                    "type": "icu_collation_keyword",
                    "index": False,
                    "numeric": True
                }
            }
        },
        "speaker": {
            "type": "text",
            "analyzer": "custom_text_analyzer"
        },
        "translations": {
            "type": "nested",
            "properties": {
                "lang": {
                    "type": "keyword"
                },
                "original": {
                    "type": "long"
                },
                "text": {
                    "type": "text",
                    "analyzer": "custom_text_analyzer"
                }
            }
        }
    }
}

WORDS_INDEX_SETTINGS = {
    "index.max_inner_result_window": 10000,
    "index.max_result_window": 10000,
    "index.number_of_replicas": 2,
    "index.refresh_interval": "30s",
    "analysis": {
        "filter": {
            "custom_ascii_folding": {
                "type": "asciifolding",
                "preserve_original": "false"
            }
        },
        "analyzer": {
            "custom_text_analyzer": {
                "filter": [
                    "custom_ascii_folding",
                    "lowercase"
                ],
                "type": "custom",
                "tokenizer": "standard"
            }
        }
    }
}
WORDS_INDEX_MAPPING = {
    "properties": {
        "coordinates": {
            "properties": {
                "page": {
                    "type": "long"
                },
                "x0": {
                    "type": "float"
                },
                "x1": {
                    "type": "float"
                },
                "y0": {
                    "type": "float"
                },
                "y1": {
                    "type": "float"
                }
            }
        },
        "lang": {
            "type": "keyword"
        },
        "lemma": {
            "type": "text",
            "analyzer": "custom_text_analyzer"
        },
        "meeting_id": {
            "type": "keyword",
            "fields": {
                "sort": {
                    "type": "icu_collation_keyword",
                    "index": False,
                    "numeric": True
                }
            }
        },
        "original": {
            "type": "long"
        },
        "pos": {
            "type": "long"
        },
        "propn": {
            "type": "long"
        },
        "segment_id": {
            "type": "keyword",
            "fields": {
                "sort": {
                    "type": "icu_collation_keyword",
                    "index": False,
                    "numeric": True
                }
            }
        },
        "sentence_id": {
            "type": "keyword",
            "fields": {
                "sort": {
                    "type": "icu_collation_keyword",
                    "index": False,
                    "numeric": True
                }
            }
        },
        "speaker": {
            "type": "text",
            "analyzer": "custom_text_analyzer"
        },
        "text": {
            "type": "text",
            "analyzer": "custom_text_analyzer"
        },
        "word_id": {
            "type": "keyword",
            "fields": {
                "sort": {
                    "type": "icu_collation_keyword",
                    "index": False,
                    "numeric": True
                }
            }
        },
        "wpos": {
            "type": "long"
        },
        "join": {
            "type": "keyword"
        },
        "type": {
            "type": "keyword"
        }
    }
}

PLACES_INDEX_SETTINGS = {
    "index.number_of_replicas": 2,
    "index.refresh_interval": "30s",
}
ATTENDEES_INDEX_SETTINGS = {
    "index.number_of_replicas": 2,
    "index.refresh_interval": "30s",
}


# upload a list of elements to Elasticsearch
def upload_to_elasticsearch(elements, index_name):
    actions = []
    for element in elements:
        action = {
            "_op_type": "index",  # Index operation
            "_index": index_name,  # Index name
            "_source": element  # Document data
        }
        actions.append(action)

    # if actions list is longer than 1000, split it into multiple lists and upload them separately
    if len(actions) > 500:
        for i in range(0, len(actions), 500):
            helpers.bulk(es, actions[i:i + 500])
        print(index_name + ": Uploading " + str(len(elements)) + " elements to Elasticsearch...")
    else:
        helpers.bulk(es, actions)
        print(index_name + ": Uploading " + str(len(actions)) + " elements to Elasticsearch...")


def create_index(index_name, settings, mappings):
    if not es.indices.exists(index=index_name):
        print("Creating index: " + index_name + "\n")
        es.indices.create(index=index_name, settings=settings, mappings=mappings)
    elif es.indices.exists(index=index_name) and DELETE_INDEX_IF_EXISTS:
        print("Deleting index: " + index_name)
        es.indices.delete(index=index_name)
        print("Creating index: " + index_name + "\n")
        es.indices.create(index=index_name, settings=settings, mappings=mappings)


def main():
    # Create the Elasticsearch indices if they don't exist
    create_index(MEETINGS_INDEX_NAME, MEETINGS_INDEX_SETTINGS, MEETINGS_INDEX_MAPPING)
    create_index(SENTENCES_INDEX_NAME, SENTENCES_INDEX_SETTINGS, SENTENCES_INDEX_MAPPING)
    create_index(WORDS_INDEX_NAME, WORDS_INDEX_SETTINGS, WORDS_INDEX_MAPPING)
    create_index(PLACES_INDEX_NAME, PLACES_INDEX_SETTINGS, {})
    create_index(ATTENDEES_INDEX_NAME, ATTENDEES_INDEX_SETTINGS, {})

    files_uploaded = 0

    # Upload the data to Elasticsearch
    jsonl_files = os.listdir(PATH_TO_JSONL_FILES)
    for jsonl_file in jsonl_files:

        if MAX_FILES - files_uploaded == 0:
            break

        if files_uploaded < SKIP_FILES_TO:
            files_uploaded += 1
            continue

        if jsonl_file.endswith("_meeting.jsonl"):
            with open(PATH_TO_JSONL_FILES + "/" + jsonl_file, "r", encoding="utf-8") as file:
                meetings = file.readlines()
                upload_to_elasticsearch(meetings, MEETINGS_INDEX_NAME)
        elif jsonl_file.endswith("_sentences.jsonl"):
            with open(PATH_TO_JSONL_FILES + "/" + jsonl_file, "r", encoding="utf-8") as file:
                sentences = file.readlines()
                upload_to_elasticsearch(sentences, SENTENCES_INDEX_NAME)
        elif jsonl_file.endswith("_words.jsonl"):
            with open(PATH_TO_JSONL_FILES + "/" + jsonl_file, "r", encoding="utf-8") as file:
                words = file.readlines()
                upload_to_elasticsearch(words, WORDS_INDEX_NAME)
        elif jsonl_file == "krajevna_imena.jsonl":
            with open(PATH_TO_JSONL_FILES + "/" + jsonl_file, "r", encoding="utf-8") as file:
                krajevna_imena = file.readlines()
                upload_to_elasticsearch(krajevna_imena, PLACES_INDEX_NAME)
        elif jsonl_file == "poslanci.jsonl":
            with open(PATH_TO_JSONL_FILES + "/" + jsonl_file, "r", encoding="utf-8") as file:
                poslanci = file.readlines()
                upload_to_elasticsearch(poslanci, ATTENDEES_INDEX_NAME)
        else:
            print("uploading: " + jsonl_file + " failed")
            continue

        files_uploaded += 1
        print("uploaded: " + jsonl_file)
        print("progress: " + str(files_uploaded) + "/" + str(len(jsonl_files)) + "\n")

    print("Uploaded meetings, sentences and words to Elasticsearch")


if __name__ == "__main__":
    main()
