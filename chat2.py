# TODO: Forgetting (merging/deleting vectors that are similar to each other)
# TODO: Weighting of different memories (how?)
# TODO: considering negative prompts. Is that possible?

import logging
import os
import openai
import json
import re
import datetime
import pinecone
from pathlib import Path
from arxiv import arxiv
from time import time,sleep
from uuid import uuid4
from utils import parse_pdf

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


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'SciHelper:']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()
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
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
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


def load_conversation(results):
    result = list()
    for m in results['matches']:
        info = load_json('nexus/%s.json' % m['id'])
        result.append(info)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip()


def fetch_arxiv_paper(paper_id: str):
    response = arxiv.Search(id_list=[paper_id],
                            max_results=1,
                            sort_by=arxiv.SortCriterion.LastUpdatedDate,
                            sort_order=arxiv.SortOrder.Descending)
    if len(response.id_list) >= 1:
        return next(response.results())
    else:
        LOGGER.warning("There is no article with that id!")
        return None


def initialize_conversation(vdb, init_json_path="nexus/user-init.json"):
    json_path = Path(init_json_path)
    id = init_json_path.split()
    if json_path.exists():
        print("Welcome back! My name is SciHelper. I am here to help you with your research. Let's continue our adventure!\n")
        message = json.load(json_path.open("r"))["message"]
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
        vdb.upsert([(id, vector)])  # upload to Pinecone

    # send init to GPT API
    output = gpt3_completion(message)
    LOGGER.info(output)


def initialize_article(vdb, article_id):
    article_folder = Path("articles/")
    if article_folder.joinpath(f"{article_id}/metadata.json").exists():
        metadata = json.load(article_folder.joinpath(f"{article_id}.json").open("r"))
        # TODO: load vectors connected to the article from vdb
    else:
        paper = fetch_arxiv_paper(article_id)
        short_id = paper.get_short_id()
        metadata = {"id": short_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "categories": paper.categories,
                    "published": paper.published,
                    "journal_ref": paper.journal_ref,
                    "links": paper.links,
                    "url": paper.pdf_url,
                    "summary": paper.summary}
        specific_folder = article_folder.joinpath(short_id)
        json.dump(metadata, specific_folder.joinpath("metadata.json").open("w"), indent=4)
        LOGGER.info(f"Downloading pdf from ArXiv for article with id: {short_id}")
        paper.download_pdf(dirpath=specific_folder, filename="paper.pdf")
        LOGGER.info("Download complete")

    # TODO: parse from arxiv source files instead? Maybe into HTML with arxiv-vanity
    text, _ = parse_pdf(article_folder.joinpath(metadata["id"]).joinpath("paper.pdf"))
    prompt = open_file('prompt_article.txt').replace('<<ARTICLE>>', text)

    # TODO: continue here,
    #   we will need to send "prompt" through GPT and save the output
    #   we should vectorize the prompt and save it in pinecone for information retrieval



def conversation_turn(speaker: str, message: str, payload: list[tuple[str, float]]):
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
        results = vdb.query(vector=vector, top_k=convo_length)
        conversation = load_conversation(results)  # results should be a DICT with 'matches' which is a LIST of DICTS, with 'id'
        prompt = open_file('prompt_response.txt').replace('<<CONVERSATION>>', conversation).replace('<<MESSAGE>>', message)
        # generate response, vectorize, save, etc
        output = gpt3_completion(prompt)

    return output, vector, metadata


if __name__ == '__main__':
    convo_length = 5
    openai.api_key = open_file('key_openai.txt')
    pinecone.init(api_key=open_file('key_pinecone.txt'), environment='us-east-1-aws')
    vdb = pinecone.Index("llm-ltm")

    # ask for user bio and preferences, if they don't exist already
    initialize_conversation(vdb)

    # initialize article
    input("Please provide valid ArXiv article ID: ") # TODO: support full urls as well
    initialize_article(vdb)
    # TODO: this should be repeatable (maybe different thread and invoke at specific USER prompt?)
    #   when we have web-ui, we can use a button for this

    while True:
        #### get user input, save it, vectorize it, save to pinecone
        payload = list()
        # USER turn
        message = input('\n\nUSER: ')
        output, _, _ = conversation_turn("USER", message, payload)

        # SciHelper turn
        _, _, _ = conversation_turn("SciHelper", output, payload)
        vdb.upsert(payload)
        print('\n\nSciHelper: %s' % output)