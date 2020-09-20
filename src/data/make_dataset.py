# -*- coding: utf-8 -*-
import os
import sys
import random
import pandas as pd
from tqdm import tqdm

def read_file(path):
    """read files from path

    Args:
        path (str): full path of the file

    Returns:
        list(str): list of lines
    """
    with open(path, "r", encoding="utf8") as f:
        lines = f.readlines()
        f.close()
    return lines


def get_highlights(lines, split_on='@highlight'):
    """get summary highlights of a story
       for example - 
        "
                (CNN) -- If you travel by plane and arriving on time makes a difference, try to book on Hawaiian Airlines. In 2012, passengers got where they needed to go without delay on the carrier more than nine times out of 10, according to a study released on Monday.

                In fact, Hawaiian got even better from 2011, when it had a 92.8% on-time performance. Last year, it improved to 93.4%.

                [...]

                @highlight

                Hawaiian Airlines again lands at No. 1 in on-time performance

                @highlight

                The Airline Quality Rankings Report looks at the 14 largest U.S. airlines

                @highlight

                ExpressJet and American Airlines had the worst on-time performance

                @highlight

                Virgin America had the best baggage handling; Southwest had lowest complaint rate
        "

    Args:
        lines (list(str)): lines of a story
        split_on (str, optional): do not change. Defaults to '@highlight'.

    Returns:
        list(str): list of highlight sentences
    """
    text = " ".join(lines)
    splits = text.split(split_on)
    return splits[:1], splits[1:]


class CleanText:
    """basic cleaning of text (lowercasing and removing special chars)
    """
    def __init__(self):
        pass

    def lower(self, text_list):
        """lower list of strings

        Args:
            text_list (list(str)): data

        Returns:
            list(str): lowered data
        """
        return [text.lower() for text in text_list]

    @staticmethod
    def _remove_special_chars(sentence, replace_with=""):
        """remove special characters (if any)

        Args:
            sentence (str): sentence
            replace_with (str, optional): replace with this string. Defaults to "".

        Returns:
            str: sentence without \n and  \t
        """
        sentence = sentence.replace('\n', replace_with).replace('\t', replace_with)
        return sentence

    def remove_special_chars(self, text_list):
        """implemented _remove_special_chars on a list of string

        Args:
            text_list (list(str)): list of strings

        Returns:
            list(Str): processed list of strings
        """
        return [self._remove_special_chars(text) for text in text_list]


if __name__ == '__main__':
    DATA_PATH = "../../data/processed/cnn_stories_tokenized"
    FILE_PATHS = os.listdir(DATA_PATH)

    print("{} files detected!".format(len(FILE_PATHS)))

    df = {
        'text': [],
        'summary': []
    }

    cleaner = CleanText()
    for path in  tqdm(FILE_PATHS[:100]):
        full_path = os.path.join(DATA_PATH, path)
        
        lines = read_file(full_path)
        lines, highlights = get_highlights(lines)

        lines = cleaner.lower(lines)
        lines = cleaner.remove_special_chars(lines)
        text = " ".join(lines)

        highlights = cleaner.lower(highlights)
        highlights = cleaner.remove_special_chars(highlights)

        summary = ". ".join(highlights)

        df['text'].append(text)
        df['summary'].append(summary)

    DF_PATH = "../../data/processed/data.csv"

    df = pd.DataFrame(df)

    print(df.head())

    df[['text', 'summary']].to_csv(DF_PATH)

