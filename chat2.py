# DONE: save article sections locally for fetching (in metadata.json)
# DONE: save and fetch vectors via Pinecone ()
# DONE: MIGRATE to ChatCompletion instead of Completion to save API tokens
# TODO:0 !!! if section_content longer than some maximum, split it into two and recursively run with summary of the previous sections attached
# TODO:0 @line215 TypeError: object of type 'UUID' has no len()

# TODO:1 Forgetting (merging/deleting vectors that are similar to each other)
# TODO:1 Weighting of different memories (how?)
# TODO:1 considering negative prompts. Is that possible?
# DONE:1 split papers into multiple smaller chunks (vectorize those and save into files for retrieval)

# TODO:3 use local Alpaca instead of ChatGPT (https://github.com/tloen/alpaca-lora)
import logging
import os

import openai
import json
import re
import datetime
import pinecone
from pathlib import Path
from time import time, sleep
from uuid import uuid4
from utils import cleanup_prompt, extract_text_from_completion_api_response, extract_text_from_chat_api_response, \
    fetch_and_download_from_arxiv, extract_arxiv_id, choose_source_file_parser
from flags import NS, N_TOP_ARTICLES, N_TOP_CONVOS, ARTICLE_DIR, LOG_DIR

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.0, top_p=1.0, tokens=1000, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'SciHelper:']):
    max_retry = 5
    retry = 0
    prompt = cleanup_prompt(prompt)
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = extract_text_from_completion_api_response(response)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def gpt3_chat_completion(conversation: list[dict[str: str]], model="gpt-3.5-turbo", log_folder=LOG_DIR) -> list[dict]:
    """ provide completion output from chat-gpt api given a list of messages (dicts) in the current conversation

    :param conversation: list of conversation turns (dicts).
        keys are "role" (:"system", "assistant" or "user") and "content" (: message_string)

        example conversation input:
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
    :param model: (str) name of openai gpt api model to use for completion
    :param log_folder: (str or Path) path to log folder for conversations

    :return: updated conversation list of message dictionaries with new message included
    """
    max_retry = 5
    retry = 0

    # NOTE: dont forget to clean up the user prompts before putting them into conversation dict

    log_fldr = Path(log_folder)
    log_fldr.mkdir(exist_ok=True)

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=conversation,
            )
            role, content = extract_text_from_chat_api_response(response)
            conversation.append({"role": role, "content": content})
            filename = f'{time()}_chatgpt3.json'
            json.dump(conversation, log_fldr.joinpath(filename).open('w'), indent=4)
            return conversation
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def load_conversation(results):
    result = list()
    for m in results['matches']:
        info = load_json('nexus/%s.json' % m['id'])
        result.append(info)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip()


def load_articles(results):
    top_k_list = list()
    for m in results['matches']:
        aid = m["id"]
        top_k_list.append(aid)
    return top_k_list


def initialize_conversation(vdb, init_json_path="nexus/user-init.json"):
    assert init_json_path.endswith(".json")
    json_path = Path(init_json_path)
    id = json_path.name.removesuffix(".json")
    if json_path.exists():
        print("Welcome back! My name is SciHelper. I am here to help you with your research. Let's continue our adventure!\n")
        metadata = json.load(json_path.open("r"))
        vector = vdb.fetch([id], namespace=NS.USER_INIT.value)
        # TODO: add user-bio reset option
    else:
        print('\nWelcome, my name is SciHelper. I am here to help you with your research. And you are?\n')
        user_bio = input('Please provide a short background on and preferred areas of your research:\n')
        print("\n\nThank you! I will remember that!")
        user_goals = input("\nI'd also love to have a list of goals or expectations of your research:\n")
        print("\n\nThank you! Now let's research something wonderful together!")
        message = open_file('prompt_init.txt').replace('<<USER_BIO>>', user_bio).replace('<<USER_GOALS>>', user_goals)

        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        # message = '%s: %s - %s' % ('USER', timestring, a)
        vector = gpt3_embedding(message)
        metadata = {'speaker': 'INIT', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': id}
        save_json(json_path, metadata)
        vdb.upsert([(id, vector)], namespace=NS.USER_INIT.value)  # upload to Pinecone

    # # send init to GPT API
    # output = gpt3_completion(message)
    # LOGGER.info(output)

    return vector, metadata


def initialize_article(vdb, article_id, article_folder=ARTICLE_DIR, use_chat_api=True):
    vectors = []
    path_to_metadata = article_folder.joinpath(f"{article_id}/metadata.json")
    if path_to_metadata.exists():
        metadata = json.load(path_to_metadata.open("r"))
    else:
        source_file_path, metadata = fetch_and_download_from_arxiv(article_id, article_folder)
        parser = choose_source_file_parser(source_file_path)
        sections = parser(source_file_path)

        # TODO: utilize ASYNC/threading for multiple parallel requests as this is very slow
        # summarize article sections and save the vector representations into Pinecone
        for section_title, section_content in sections.items():
            # compress article sections into concise summaries
            prompt = open_file('prompt_summarize.txt').replace('<<TEXT>>', f"{section_title}\n{section_content}")
            if use_chat_api:
                # use chat completion api (chatgpt)
                # TODO:0 !!! if section_content longer than some maximum, split it into two and recursively run with summary of the previous sections attached
                conversation = [{"role": "system", "content": cleanup_prompt(prompt)}, ]
                LOGGER.debug(conversation)
                summary = gpt3_chat_completion(conversation)[-1]["content"]
            else:
                # use text completion api (davinci)
                summary = gpt3_completion(prompt).strip()
            vector_id = uuid4()
            vectors.append((vector_id, gpt3_embedding(summary), {"article": article_id, "title": section_title}))
            metadata['section_summaries'][vector_id] = (section_title, summary)
            # TODO: as we have multiple vectors for one article, query by filtering article_id or title in metadata

        vdb.upsert(vectors, namespace=NS.ARTICLES.value)
        json.dump(metadata, path_to_metadata.open("w"))

    return metadata


def conversation_turn(speaker: str, message: str, payload: list[tuple[str, float]],
                      n_top_convos=N_TOP_CONVOS, n_top_articles=N_TOP_ARTICLES):
    timestamp = time()
    timestring = timestamp_to_datetime(timestamp)

    unique_id = str(uuid4())
    metadata = {'speaker': speaker, 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id}
    save_json('nexus/%s.json' % unique_id, metadata)
    vector = gpt3_embedding(message)
    payload.append((unique_id, vector))

    if "SciHelper" in speaker:
        output = None
    else:
        # search for relevant messages, and generate a response
        prev_convos = vdb.query(vector=vector, top_k=n_top_convos, namespace=NS.CONVOS.value)
        prev_articles = vdb.query(vector=vector, top_k=n_top_articles, namespace=NS.ARTICLES.value)
        conversation = load_conversation(prev_convos)  # results should be a DICT with 'matches' which is a LIST of DICTS, with 'id'
        article_ids = load_articles(prev_articles)  # TODO: now what do we actually do with these?, we should cache the article texts I guess
        prompt = open_file('prompt_response.txt').replace('<<CONVERSATION>>', conversation).replace('<<MESSAGE>>', message)
        # generate response, vectorize, save, etc
        output = gpt3_completion(prompt)

    return output, vector, metadata


def prompt_for_article():
    while True:
        usr_input = input("Please give me a valid ArXiv article address or ID: ")

        article_id = extract_arxiv_id(usr_input)  # TODO async: start downloading immediately

        if article_id:
            print("Thank you!")
            return article_id
        else:
            print("I couldn't find any article id in your input. Can you recheck and try again?\n")
            # TODO: "Would you like to continue without a specific article?"


if __name__ == '__main__':
    openai.api_key = open_file('key_openai.txt')
    pinecone.init(api_key=open_file('key_pinecone.txt'), environment='us-east-1-aws')
    vdb = pinecone.Index("llm-ltm")

    # ask for user bio and preferences, if they don't exist already
    initialize_conversation(vdb)

    # initialize article
    article_id = prompt_for_article()
    metadata = initialize_article(vdb, article_id)
    # NOTE: this should be repeatable (maybe different thread and invoke at specific USER prompt?)
    #   when we have web-ui, we can use a button for this

    print(metadata)

    # fetch all vectors for given article
    vectors = vdb.fetch([metadata["section_summaries"].keys()], namespace=NS.ARTICLES.value)
    print(vectors)
    exit()

    # TODO: utilize summarized article in further user queries
    #   - NOTE: maybe summarize the summaries one more time for ultimate summary that saves us api tokens

    while True:
        #### get user input, save it, vectorize it, save to pinecone
        payload = list()
        # USER turn
        message = input('\n\nUSER: ')
        output, _, _ = conversation_turn("USER", message, payload)

        # SciHelper turn
        _, _, _ = conversation_turn("SciHelper", output, payload)
        vdb.upsert(payload, namespace=NS.CONVOS.value)
        print('\n\nSciHelper: %s' % output)