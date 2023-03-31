import logging
import os
from collections import defaultdict

import openai
import json
import jsonlines
from datetime import datetime
import pinecone
from pathlib import Path
from time import time, sleep
from uuid import uuid4
from utils import handle_flag, cleanup_prompt, extract_text_from_completion_api_response, \
    extract_text_from_chat_api_response, fetch_and_download_from_arxiv, extract_arxiv_id, choose_source_file_parser, \
    pineconize, depineconize, open_file, save_file, save_json, timestamp_to_datetime, num_tokens_in_convo, split_text_to_n_token_chunks
from flags import NS, N_TOP_SECTIONS, N_TOP_CONVOS, ARTICLE_DIR, TEMPLATES_DIR, LOG_DIR, NEXUS_DIR, TIME_FORMAT, \
    N_MAX_WORDS_IN_CONV, \
    DEFAULT_EMBEDDING_ENGINE, DEFAULT_MODEL, GENERATE_SUMMARY, Commands, CONVO_SCORE_CUTOFF, SECTION_SCORE_CUTOFF

logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger(__name__)


def gpt3_embedding(content: str, engine=DEFAULT_EMBEDDING_ENGINE):
    content = content.encode(encoding='ASCII', errors='ignore').decode()  # fix any UNICODE errors
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def conversation2string(convo: list[dict[str: str, str: str]]) -> str:
    return '\n\n'.join([f"{c['role']}:\n{c['content']}" for c in convo])


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.0, top_p=1.0, tokens=1000, freq_pen=0.0, pres_pen=0.0,
                    stop=['USER:', 'SciHelper:']):
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


def gpt3_chat_completion(conversation: list[dict[str: str]], model=DEFAULT_MODEL, log_folder=LOG_DIR) -> list[dict]:
    """ Provide completion output from chat-gpt api given a list of messages (dicts) in the current conversation.

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
            filename = f'{datetime.now().strftime(TIME_FORMAT)}_chatgpt3.jsonl'
            with jsonlines.open(log_fldr.joinpath(filename), mode="a") as writer:
                writer.write_all(conversation[-2:])
            return conversation
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            LOGGER.warning('Error communicating with OpenAI:', oops)
            sleep(1)


def load_conversation(matches: list[str], nexus_folder=NEXUS_DIR) -> list[dict[str: str]]:
    """

    :param matches: results from vdb query  for convos (list of strings which are ids of the convos)
    :param nexus_folder: folder in which conversations are stored
    :return: list[dict["role": str, "content": str]] ... joined list of conversations
             [] if no matches found
    """
    nexus_folder = Path(nexus_folder)
    convo_list = list()
    for m in matches:
        print(m)
        try:
            with jsonlines.Reader(nexus_folder.joinpath(f"{m}.jsonl").open("r")) as reader:
                info = reader.iter()
                convo_list.extend(info)
        except FileNotFoundError as err:
            LOGGER.warning(f"(skipping) m in matches@load_conversation: {err}")
            continue
        except KeyError as err:
            LOGGER.warning(f"(skipping) m in matches@load_conversation: KeyError({err})")
            continue
    LOGGER.debug(f"convo_list@load_conversation: {convo_list}")
    return convo_list


def load_articles(results):
    top_k_list = list()
    for m in results['matches']:
        aid = m["id"]
        top_k_list.append(aid)
    return top_k_list


def initialize_user(vdb, init_json_path="nexus/user-init.json", role='system'):
    assert init_json_path.endswith(".json")
    json_path = Path(init_json_path)
    id = json_path.name.removesuffix(".json")
    if json_path.exists():
        print(
            "Welcome back! My name is SciHelper. I am here to help you with your research. Let's continue our adventure!\n")
        user_metadata = json.load(json_path.open("r"))
        vector = vdb.fetch([id], namespace=NS.USER_INIT.value)
        # TODO: add user-bio reset option
    else:
        print('\nWelcome! My name is SciHelper. I am here to help you with your research. And you are?\n')
        user_bio = input(
            'Please introduce yourself and your research background as well as preferred research areas:\n')
        print("\n\nThank you! I will remember that!")
        user_goals = input("\nAnd one more thing. What are the aims and goals of your current research?\n")
        print("\n\nThank you! Now let's research something wonderful together!")
        message = open_file(TEMPLATES_DIR.joinpath('prompt_init.txt')).replace('<<USER_BIO>>', user_bio).replace('<<USER_GOALS>>', user_goals)

        conversation = gpt3_chat_completion([{"role": role, "content": message}])
        summary = conversation[-1]["content"]

        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        # message = '%s: %s - %s' % ('USER', timestring, a)
        vector = gpt3_embedding(message)
        user_metadata = {'role': role, 'time': timestamp, 'bio': summary, 'timestring': timestring, 'uuid': id}
        save_json(json_path, user_metadata)
        vdb.upsert([(id, vector)], namespace=NS.USER_INIT.value)  # upload to Pinecone

    # # send init to GPT API
    # output = gpt3_completion(message)
    # LOGGER.info(output)

    return vector, user_metadata


def initialize_article(vdb: pinecone.Index, article_id: str, article_folder=ARTICLE_DIR, use_chat_api=True,
                       generate_summary=False) -> (dict, dict):
    """ Load existing article or initialize a new article based on article_id from arXiv.org.
    If initializing new, the article is divided into sections, summarized one by one by GPT and then saved into
    metadata['section_summaries'] with unique vector_ids. Sections are then vectorized and upserted to pinecone vdb
    with the same vector_ids for fetching. pinecone vectors also have metadata with piconized(article_id) for
    fetching article texts from local metadata.json file when semantic similarity searching.

    :param vdb: (pinecone.Index) pinecone vector database index object
    :param article_id: (str) arXiv id of an article
    :param article_folder: (str) local folder where articles are stored (flags.ARTICLE_DIR)
    :param use_chat_api: (bool) if true, openai.ChatCompletion.create is used, else Completion.create is used
    :param generate_summary: (bool: False) if true, abstract is created by using GPT3/ if false, original abstract is used
    :return: (dict, dict) metadata dictionary from local files and vectors dictionary fetched from pinecone
    """
    path_to_metadata = article_folder.joinpath(f"{article_id}/metadata.json")
    if path_to_metadata.exists():
        metadata = json.load(path_to_metadata.open("r"))
    else:
        source_file_path, metadata = fetch_and_download_from_arxiv(article_id, article_folder)
        parser = choose_source_file_parser(source_file_path)
        sections = parser(source_file_path)
        vector_list = []

        # TODO: utilize ASYNC/threading for multiple parallel requests as this is very slow
        # summarize article sections and save the vector representations into Pinecone
        for section_title, section_content in sections.items():
            # compress article sections into concise summaries
            section_chunks = split_text_to_n_token_chunks(f"{section_title}\n{section_content}")
            # TODO: can be made much better. e.g. split by periods and then approximately split at the closest sentence end.
            summary = None
            for chunk in section_chunks:
                if summary:
                    chunk = f"{summary}\n\n{chunk}"  # feedback loop to include summaries of previous chunks as well.
                prompt = open_file(TEMPLATES_DIR.joinpath('prompt_summarize.txt')).replace('<<TEXT>>', chunk)
                if use_chat_api:
                    # use chat completion api (chatgpt)
                    # TODO:0 !!! if section_content longer than some maximum, split it into two and recursively run with summary of the previous sections attached
                    conversation = [{"role": "system", "content": cleanup_prompt(prompt)}, ]
                    LOGGER.debug(conversation)
                    summary = gpt3_chat_completion(conversation)[-1]["content"]
                else:
                    # use text completion api (davinci)
                    summary = gpt3_completion(prompt).strip()
            vector_id = uuid4().hex
            # NOTE: pinecone identifies strings with . as DATETIME type ... uff, using pineconize fun
            vector_list.append((vector_id, gpt3_embedding(summary), {"article": pineconize(article_id),
                                                                     "title": section_title}))
            metadata['section_summaries'][vector_id] = (section_title, summary)
            LOGGER.info(f"------\n{metadata['section_summaries'][vector_id][0]}\n{metadata['section_summaries'][vector_id][1]}\n------\n\n")
            # NOTE: as we have multiple vectors for one article, query by filtering article_id or title in metadata

        # article-wide summary (aka custom abstract)
        if generate_summary:
            section_summaries = "\n\n".join([f"{c[0]}\n{c[1]}" for c in metadata['section_summaries']])
            user_bio = json.load(NEXUS_DIR.joinpath("user-init.json").open("r"))['bio']
            message = open_file(TEMPLATES_DIR.joinpath('prompt_article-wide_summary.txt')).replace("<<BIO>>",
                                                                                        user_bio).replace("<<TEXT>>",
                                                                                             section_summaries)
            article_summary = gpt3_chat_completion([{"role": "user", "content": message}])[-1]  # only take the summary
            metadata['summary'] = article_summary["content"]

        vdb.upsert(vector_list, namespace=NS.ARTICLES.value)
        json.dump(metadata, path_to_metadata.open("w"))

    vectors = vdb.fetch(list(metadata["section_summaries"].keys()), namespace=NS.ARTICLES.value)

    return metadata, vectors


def fetch_similar(vdb: pinecone.Index, vector, namespace=NS.ARTICLES.value, top_k=50, score_cutoff=0.5,
                  include_values=False, include_metadata=False, excluded_ids: set = tuple()) -> list:
    """ Fetches maximum of 'top_k' number of article sections (or other vectors) with similar semantic meaning.
    If similarity score is < score_cutoff, then the respective article section is not included in results.


    :param vdb: (pinecone.Index) pinecone vector database index object
    :param vector: (list[float]) vector by which we search similar ones, list of float values
    :param namespace: (str: NS.ARTICLES.value) index namespace to search within
    :param top_k: (int: 50)maximum number of results found
    :param score_cutoff: (float: 0.5) similarity score threshold, if score is lower than cutoff, respective match is discarded
    :param include_values: (bool: False), whether match should include vector values
    :param include_metadata: (bool: False), whether match should include metadata
    :param excluded_ids: (set: {}) vector ids that should not be fetched
    :return: (list[dict]) matches without excluded_ids and with similarity score above the cutoff in a list of dictionaries
     example output: [
        {
          "id": str,
          "score": score_cutoff < float < 1,
          "values": [float],
          "sparseValues": {
            "indices": [int, ...],
            "values": [float, ...]
          },
          "metadata": {str: str}
        },
        ...
     ]
    """
    response = vdb.query(namespace=namespace,
                         top_k=top_k,
                         include_values=include_values,
                         include_metadata=include_metadata,
                         vector=vector)
    filtered_matches = []
    for match in response['matches']:
        score = match['score']
        if match['id'] not in excluded_ids and score >= score_cutoff:
            filtered_matches.append(match)
    print(filtered_matches)

    return filtered_matches


def fetch_article_sections(article_id: str, vector_ids: list[str]) -> list[tuple[str, str]]:
    """ fetch specific section with 'vector_id' from article with 'article_id'

    :param article_id: (str) id of a containing article
    :param vector_ids: list[str] ids of the specific sections we want to fetch
    :return: (section_title, summary)
    """
    try:
        metadata = json.load(ARTICLE_DIR.joinpath(depineconize(article_id)).joinpath("metadata.json").open("r"))
        return [tuple(metadata['section_summaries'][vid]) for vid in vector_ids]
    except FileNotFoundError as err:
        LOGGER.warning(f"{err}. Returning empty list")
        return []


def new_conversation(user_bio: str, nexus_folder=NEXUS_DIR):
    nexus_folder.mkdir(exist_ok=True)

    timestring = datetime.now().strftime(TIME_FORMAT)
    conv_id = f"{timestring}_{uuid4().hex}"

    conversation = [{"role": "system", "content": user_bio}]

    with jsonlines.open(nexus_folder.joinpath(f"{conv_id}.jsonl"), mode="w") as writer:
        writer.write_all(conversation)

    return conv_id, conversation, timestring


def _summarize_conversation(conversation: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Summarizes a conversation if it exceeds the maximum number of words allowed.
    The first ("system") message is left unchanged as well as the last ("user") message.

    :param conversation: list[dict[str, str]], the full conversation to be summarized
    :return: list[dict[str, str]], the updated conversation (either the original or the summarized version)
    """
    total_length = num_tokens_in_convo(conversation, DEFAULT_MODEL)
    LOGGER.info(f"conversation length: {total_length}")
    if total_length < N_MAX_WORDS_IN_CONV:
        return conversation

    LOGGER.info(f"conversation is too long (n_tokens: {total_length})... Summarizing.")
    LOGGER.info(f"conversation (before summary): {conversation}")
    fc_str = {"role": "system", "content": open_file(TEMPLATES_DIR.joinpath('prompt_conv_compression.txt')).replace(
        '<<FULL_CONVERSATION>>', conversation2string(conversation[1:-1]))}
    conv_summary = gpt3_chat_completion([fc_str])[-1:]
    conversation_summary = conversation[:1] + conv_summary + conversation[-1:]  # adding first and last message
    return conversation_summary


def conversation_turn(speaker: str, conversation: list[dict[str, str]], conv_id: str, vdb: pinecone.Index,
                      n_top_convos=N_TOP_CONVOS, n_top_sections=N_TOP_SECTIONS,
                      nexus_folder=NEXUS_DIR, excluded_conv_ids: set = tuple(), excluded_vids: set = tuple()):
    """
    Generates a response based on the current conversation and metadata about previous conversations and articles.

    :param speaker: str, the current speaker
    :param conversation: list[dict[str, str]], the current conversation
    :param conv_id: str, the conversation ID
    :param n_top_convos: int, the number of top conversations to fetch from the database
    :param vdb: pinecone.Index, object connection to the pinecone vector database index
    :param n_top_sections: int, the number of top sections to fetch from the database
    :param nexus_folder: str, the folder where the conversation is saved
    :param excluded_conv_ids: set, vector ids of saved conversations which should be excluded from fetching (most likely already fetched in previous turns)
    :param excluded_vids: set, vector ids of article sections which should be excluded from fetching (most likely already fetched in previous turns)
    :return: list[dict[str, str]], the updated conversation
    """

    excluded_conv_ids = set(excluded_conv_ids)
    excluded_vids = set(excluded_vids)
    if "SciHelper" in speaker:
        output = None
    else:

        vector = gpt3_embedding(conversation2string(conversation[1:]))

        # Search for relevant messages and generate a response
        prev_convo_ids = [match['id'] for match in fetch_similar(vdb, vector, NS.CONVOS.value, top_k=n_top_convos,
                                                                 score_cutoff=CONVO_SCORE_CUTOFF,
                                                                 excluded_ids=excluded_conv_ids)]
        prev_conversation = load_conversation(prev_convo_ids)  # list[dict["role": str, "content": str], dict...]
        excluded_conv_ids.update(prev_convo_ids)  # update already used conversations

        # Fetch and add relevant article sections to the conversation
        # NOTE: alternatively fetch only based on the latest user input?
        prev_unfetched_sections = [(match['id'], match['metadata']['article']) for match in
                                   fetch_similar(vdb, vector, NS.ARTICLES.value, top_k=n_top_sections,
                                                 score_cutoff=SECTION_SCORE_CUTOFF, include_metadata=True,
                                                 excluded_ids=excluded_vids)]
        section_dict = defaultdict(list)
        for vid, aid in prev_unfetched_sections:
            section_dict[aid].append(vid)
            excluded_vids.add(vid)
        sections = []
        for aid, vid_list in section_dict.items():
            sections.extend(fetch_article_sections(aid, vid_list))  # list[(title, summary), ...]
        section_conversation = [{"role": "assistant", "content": "\n".join(s)} for s in sections]

        # Combine all previous messages, relevant article sections, and current message
        conversation = conversation[:1] + prev_conversation + section_conversation + conversation[1:]

        # If conversation is too long, summarize it
        conversation = _summarize_conversation(conversation)

        # Generate response, vectorize, save, etc
        conversation = gpt3_chat_completion(conversation)
        # TODO: save as json with info about convos and article sections used for this convo
        #   prev_convo_ids, vid_set
        # TODO: embed and upsert to vdb

        # conversation.append(full_conversation[-1])

    # Update conversation jsonl file
    with jsonlines.open(nexus_folder.joinpath(f"{conv_id}.jsonl"), mode="a") as writer:
        writer.write_all(conversation[-2:])

    return conversation, excluded_conv_ids, excluded_vids


def prompt_for_article():
    while True:
        usr_input = input("Please give me a valid ArXiv article address or ID: ")

        article_id = extract_arxiv_id(usr_input)  # TODO async: start downloading immediately

        if article_id:
            print("Thank you! I'm processing the article. This could take up to 5 minutes.")
            return article_id
        else:
            print("I couldn't find any article id in your input. Can you recheck and try again?\n")
            # TODO: "Would you like to continue without a specific article?"


if __name__ == '__main__':
    openai.api_key = open_file('key_openai.txt')
    pinecone.init(api_key=open_file('key_pinecone.txt'), environment='us-east-1-aws')
    vdb = pinecone.Index("llm-ltm")

    # Boot message
    print(open_file(TEMPLATES_DIR.joinpath("boot_message.txt")))

    # ask for user bio and preferences, if they don't exist already
    user_vector, user_metadata = initialize_user(vdb)
    user_bio_summary = user_metadata["bio"]
    LOGGER.info(f"user bio saved as: {user_bio_summary}")

    # v3_out['vectors'][id]['values']  # vector values
    # v3_out['vectors'][id]['metadata']  # vector metadata

    # WIP: utilize summarized article in further user queries
    #   - NOTE: maybe summarize the summaries one more time for ultimate summary that saves us api tokens

    exit_flag = False
    reset_convo_flag = True
    new_article_flag = True
    conversation, conv_id, conv_timestring = None, None, None
    excluded_conv_ids = set()
    excluded_sect_ids = set()
    while True:
        if exit_flag:
            # TODO: make into function
            LOGGER.info("Exiting program at user request.")
            if conversation is not None:  # TODO: make_async
                print(f"Summarizing and saving previous conversation with id: {conv_id}.")
                conv_summary = _summarize_conversation(conversation)[1:]  # summarize and exclude the user bio
                summary_vector = gpt3_embedding(conversation2string(conv_summary))  # generate embedding
                # send to pinecone
                vdb.upsert([(conv_id, summary_vector, {"timestamp": conv_timestring})], namespace=NS.CONVOS.value)
            break
        # new conversation
        if reset_convo_flag:
            # save previous conversation
            if conversation is not None:  # TODO: make_async
                print(f"Summarizing and saving previous conversation with id: {conv_id}.")
                conv_summary = _summarize_conversation(conversation)[1:]  # summarize and exclude the user bio
                summary_vector = gpt3_embedding(conversation2string(conv_summary))  # generate embedding
                # send to pinecone
                vdb.upsert([(conv_id, summary_vector, {"timestamp": conv_timestring})], namespace=NS.CONVOS.value)

            # generate new conversation
            conv_id, conversation, conv_timestring = new_conversation(user_bio_summary)
            excluded_conv_ids = set()
            excluded_sect_ids = set()
            reset_convo_flag = False
            print(f"New conversation id: {conv_id}")
            continue
            # NOTE: when we have web-ui, we can use a button for this
        # add new article to the coversation
        if new_article_flag:
            # initialize article
            article_id = prompt_for_article()
            article_metadata, article_vectors = initialize_article(vdb, article_id, generate_summary=GENERATE_SUMMARY)
            print(f"Adding new article with id: {article_id}")
            # article_metadata['section_summaries'].keys() == vector_ids
            # article_metadata['section_summaries'].values() == (title, content)
            article_summary = {"role": "user", "content": open_file(TEMPLATES_DIR.joinpath("prompt_add_article.txt")
                                                                    ).replace("<<TITLE>>", article_metadata['title']
                                                                              ).replace("<<ABSTRACT>>",
                                                                                        article_metadata['summary'])}
            conversation.append(article_summary)
            new_article_flag = False
            print(f"Article loaded.")
            continue
            # NOTE: when we have web-ui, we can use a button for this

        # get user input
        print(f"Ask me anything. Like what is the article about or connections to our previous conversations.")
        message = input('\n\nUSER: ')
        conversation.append({"role": "user", "content": message.strip()})

        # HANLDERS:
        exit_flag = handle_flag(conversation, Commands.EXIT)
        reset_convo_flag = handle_flag(conversation, Commands.RESET_CONVO)
        new_article_flag = handle_flag(conversation, Commands.ADD_ARTICLE)

        conversation, excluded_conv_ids, excluded_sect_ids = conversation_turn("USER", conversation, conv_id, vdb,
                                                                               excluded_conv_ids=excluded_conv_ids,
                                                                               excluded_vids=excluded_sect_ids)

        print(conversation[-1]["content"])
