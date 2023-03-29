from utils import split_text_to_n_token_chunks


def test_split_text_to_n_token_chunks():
    # Test case 1: Empty text
    text = ''
    result = split_text_to_n_token_chunks(text)
    print(result)
    assert result == ['']

    # Test case 2: Text that fits into one chunk
    text = 'This is a short text.'
    result = split_text_to_n_token_chunks(text)
    print(result)
    assert result == [text]

    # Test case 3: Text that needs to be split into multiple chunks
    text = 'This is a longer text that needs to be split into multiple chunks. It has more than the maximum number of tokens allowed in each chunk, so it should be split accordingly.'
    result = split_text_to_n_token_chunks(text, n_max_tokens=10)
    print(result)
    assert len(result) == 5
    assert len(result[0].split()) <= 10
    assert len(result[-1].split()) <= 10
    assert sum([len(chunk) for chunk in result]) == len(text)

    # Test case 4: Text with non-alphanumeric characters
    text = 'This is a text with non-alphanumeric characters: !@#$%^&*()_+{}|:"<>?,./;\'[].'
    result = split_text_to_n_token_chunks(text)
    print(result)
    assert result == [text]

    # Test case 5: Text with non-ASCII characters
    text = 'This is a text with non-ASCII characters: éáíóúàèìòùäëïöü.'
    result = split_text_to_n_token_chunks(text)
    print(result)
    assert result == [text]


if __name__ == "__main__":
    test_split_text_to_n_token_chunks()
