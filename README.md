SciHelper
========

Web-app/plugin (hopefully) for GPT-3.5. Personalised AI research
assistant with discussion recall and on demand article analysis,
summarization, storage and cross-referencing.

Features
--------

- Tailor responses based on user bio, research interests and goals
- Article assistance
  - fetch, summarize and store articles on demand
  - engage in discussions about respective articles
  - cross reference multiple articles and previous conversations about them
- Long-term Memory
  - remember gists of previous conversations
  - fetch and reference previous conversations 

Installation
------------

The project (master branch) is in alpha stage right now,
it's very slow and only works as a Python script.

To test it out: 
1. install the requirements:
```bash
pip install -r requirements.txt
```

 2. clone this repo:
```bash
git clone https://github.com/vejvarm/SciHelper.git
cd SciHelper
```

``` run main.py
python main.py
```

Contribute
----------
Feel free to put in a pull request or a suggestion.
This is very early stage and I plan on rewriting
the whole repo with LLMChain package and async support.

- Issue Tracker: github.com/vejvarm/SciHelper/issues

Support
-------

If there are any issues or suggestions, feel free to open an issue
or a pull request. Alternatively contact me directly by email in my 
profile.

License
-------

The project is licensed under the MIT license. Refer to LICENSE file.

# Acknowledgements
Big props to David Shapiro ([@daveshap](https://github.com/daveshap)), whose code for [RAVEN](https://github.com/daveshap/raven) 
was the original inspiration for this project. 
