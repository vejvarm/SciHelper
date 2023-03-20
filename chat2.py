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
from flags import NS, N_TOP_ARTICLES, N_TOP_CONVOS

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


def load_articles(results):
    top_k_list = list()
    for m in results['matches']:
        aid = m["id"]
        top_k_list.append(aid)
    return top_k_list

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
    assert init_json_path.endswith(".json")
    json_path = Path(init_json_path)
    id = json_path.name.removesuffix(".json")
    if json_path.exists():
        print("Welcome back! My name is SciHelper. I am here to help you with your research. Let's continue our adventure!\n")
        metadata = json.load(json_path.open("r"))
        message = metadata["message"]
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

    # send init to GPT API
    output = gpt3_completion(message)
    LOGGER.info(output)

    return output, vector, metadata


def initialize_article(vdb, article_id):
    article_folder = Path("articles/")
    vector = None
    if article_folder.joinpath(f"{article_id}/metadata.json").exists():
        metadata = json.load(article_folder.joinpath(f"{article_id}/metadata.json").open("r"))
        vector = vdb.fetch([article_id], namespace=NS.ARTICLES.value)
        # TODO: load vectors connected to the article from vdb
    else:
        paper = fetch_arxiv_paper(article_id)
        metadata = {"id": article_id,
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors],
                    "categories": paper.categories,
                    "published": paper.published.strftime("%Y-%M-%d"),
                    "journal_ref": paper.journal_ref,
                    "links": [l.href for l in paper.links],
                    "url": paper.pdf_url,
                    "summary": paper.summary}
        specific_folder = article_folder.joinpath(article_id)
        specific_folder.mkdir(exist_ok=True)
        json.dump(metadata, specific_folder.joinpath("metadata.json").open("w"), indent=4)
        LOGGER.info(f"Downloading pdf from ArXiv for article with id: {article_id}.\nThis may take a while ...")
        paper.download_pdf(dirpath=str(specific_folder), filename="paper.pdf")
        LOGGER.info("... download complete")

    # TODO: parse from arxiv source files instead? Maybe into HTML with arxiv-vanity
    text, _ = parse_pdf(article_folder.joinpath(metadata["id"]).joinpath("paper.pdf"))

    # save article as vector into Pinecone
    # TODO: this might be a lot of information for one vector, we should split it
    if vector is None:
        vector = gpt3_embedding(text)
        vdb.upsert([(article_id, vector)], namespace=NS.ARTICLES.value)

    # TODO: continue here,
    #   we will need to send "prompt" through GPT and save the output
    prompt = open_file('prompt_article.txt').replace('<<ARTICLE>>', text)
    output = gpt3_completion(prompt)

    return output, vector, metadata


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

        article_id = _extract_arxiv_id(usr_input)

        if article_id:
            print("Thank you!")
            return article_id
        else:
            print("I couldn't find any article id in your input. Can you recheck and try again?\n")
            # TODO: "Would you like to continue without a specific article?"


def _extract_arxiv_id(string):
    pattern = r"(?:(?:arXiv:)?([0-9]{2}[0-1][0-9]\.[0-9]{4,5}(?:v[0-9]+)?|(?:[a-z|\-]+(\.[A-Z]{2})?\/\d{2}[0-1][0-9]\d{3})))"
    # TODO: maybe ignore versions
    match = re.search(pattern, string.strip())
    if match:
        return match.group(1)
    else:
        return None


if __name__ == '__main__':
    openai.api_key = open_file('key_openai.txt')
    pinecone.init(api_key=open_file('key_pinecone.txt'), environment='us-east-1-aws')
    vdb = pinecone.Index("llm-ltm")

    # ask for user bio and preferences, if they don't exist already
    initialize_conversation(vdb)

    # initialize article
    article_id = prompt_for_article()
    initialize_article(vdb, article_id)
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
        vdb.upsert(payload, namespace=NS.CONVOS.value)
        print('\n\nSciHelper: %s' % output)