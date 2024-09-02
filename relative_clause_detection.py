from supar import Parser
import nltk
from nltk import word_tokenize
import os
import pandas as pd


class RelativeClause:

    def __init__(self, input_folder, output_folder):

        self.input_folder = input_folder
        self.output_folder = output_folder
        # make sure the path exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # load sentence segmenter
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # load dependency parser
        # https://pypi.org/project/supar/
        self.parser = Parser.load('biaffine-dep-en', reload=False)
        # load constituency parser
        self.parser2 = Parser.load('crf-con-en', reload=False)
        # load animacy dictionary
        self.animateword, self.inanimateword = self.animacy_dictionary()
        # load relativizer list
        self.relativizer_list = self.relativizer_list()

    @staticmethod
    def animacy_dictionary():
        """
        read animate and inanimate words from animacy_dictionary
        :return: list of animate words and inanimate words
        """
        animate = open(r"animacy_dictionary/animate.unigrams.txt", "r", encoding='utf-8')
        inanimate = open(r"animacy_dictionary/inanimate.unigrams.txt", "r", encoding='utf-8')
        animateword = []
        inanimateword = []
        for line in animate:
            sents = word_tokenize(line)
            for sent in sents:
                animateword.append(sent)
        for line in inanimate:
            sents = word_tokenize(line)
            for sent in sents:
                inanimateword.append(sent)
        return animateword, inanimateword

    @staticmethod
    def relativizer_list():
        """
        predefined relativizer list
        :return:
        """
        # predefined relativizer list
        relativizer_list = ["which", "whichever", "that", "who", "whom", "whose", "whoever",
                            "whomever", "why", "where", "wherever", "when", "whenever", "whence",
                            "wherein","whereon",
                            "whereof", "whereby", "what", "whatever",
                            "whatsoever", "however", "case", "than", "as", "whether" ]

        return relativizer_list

    def parsing(self, sent):
        """
        tokenization, dependency parsing and constituency parsing
        :param sent: list of words after tokenization
        :return: dataset
        """
        # word tokenization
        text = nltk.word_tokenize(sent)
        # predict dependency parsing
        dataset = self.parser.predict(text, verbose=False)
        # predict constituency parsing
        dataset2 = self.parser2.predict([text], verbose=False)

        return text, dataset, dataset2

    def acquire_subtrees(self, parse_str):
        """
        acquire all subtrees with SBAR label in the constituency parsing result
        :param parse_str: constituency parsing result
        :return: list of subtrees with SBAR label
        """
        subtrees = []
        for subtree in parse_str.subtrees():
            if subtree.label() == 'SBAR':
                subtrees.append(' '.join(subtree.leaves()))

        return subtrees

    def animacy_of_word(self, head_noun):
        """
        determine the animacy of head noun
        :return: str, animate, inanimate, unknown
        """
        if head_noun in self.animateword:
            return "animate"
        elif head_noun in self.inanimateword:
            return "inanimate"
        else:
            return "unknown"

    def zero_relative_clause(self, subtrees, clause_verb, text, index_of_head_noun):
        """
        extract zero relative clause from the sentence
        :return: str the content of zero relative clause
        """
        subtrees = list(set(subtrees))
        length_of_RC = 10000
        relative_clause_content = ''
        for subtree in subtrees:
            if clause_verb in subtree:
                if len(subtree) < length_of_RC:
                    length_of_RC = len(subtree)  # extract zero relative clause with the shortest length

                    for i in range(index_of_head_noun, len(text), 1):  # match the subtree with the text
                        if " ".join(text[i:i + len(subtree.split(" "))]) == subtree:
                            relative_clause_content = subtree

        return relative_clause_content

    def relative_clause(self, subtrees, clause_verb, relativizer, text):
        """
        extract relative clause from the sentence
        :return:
        """

        subtrees = list(set(subtrees))
        length_of_RC = 100000
        relative_clause_content = ''

        for subtree in subtrees:
            if clause_verb in subtree:
                # relativizer & clausal root in the same subtree
                if relativizer in subtree:
                    if len(subtree) < length_of_RC:
                        length_of_RC = len(subtree)  # the shortest length

                        for i in range(len(text)):  # match the subtree with the text
                            if " ".join(text[i:i + len(subtree.split(" "))]) == subtree:
                                relative_clause_content = subtree
                                break

        if length_of_RC == 100000:
            end_of_subtree, position_of_relativizer = 0, 0
            for subtree in subtrees:
                # relativizer and clausal root not in the same subtree
                if clause_verb in subtree:
                    begin_of_subtree = len(text)
                    length_of_subtree = len(subtree.split(" "))
                    # find the end_index of subtree
                    for i in range(len(text)):
                        if " ".join(text[i:i + length_of_subtree]) == subtree:
                            begin_of_subtree = i
                            end_of_subtree = i + length_of_subtree
                            break

                    # find the index of relativizer
                    for i in range(begin_of_subtree):
                        if relativizer == text[i]:
                            position_of_relativizer = i
                            break

                    if (end_of_subtree - position_of_relativizer) < length_of_RC:
                        length_of_RC = end_of_subtree - position_of_relativizer
                        relative_clause_content = " ".join(text[position_of_relativizer:end_of_subtree])

        return relative_clause_content

    def process_relative_clause(self, text, dataset, index_of_word, subtrees, i):

        relativizer_dict = {}
        rc = None

        # governing word of the clause verb, i.e., the head noun
        index_of_head_noun = dataset.arcs[i][index_of_word] - 1
        # if the relative clause is on the left side of the head noun, break
        if index_of_word < index_of_head_noun:
            return None

        # head noun
        head_noun = dataset.words[i][index_of_head_noun]
        # role of head noun in main clauses
        relation = dataset.rels[i][index_of_head_noun]

        relativizer_dict.update({"head_noun": head_noun})
        relativizer_dict.update({"head_noun_in_main_clause": relation})

        # animacy of head noun
        animacy_of_head_noun = self.animacy_of_word(head_noun)
        relativizer_dict.update({"animacy_of_head_noun": animacy_of_head_noun})

        # clausal root/verb
        # the clausal root was the word which tagged the dependency relation "rcmod".
        clause_verb = dataset.words[i][index_of_word]
        index_of_clause_verb = index_of_word
        relativizer_dict.update({"clause_verb": clause_verb})

        # distance from antecedent to clausal root
        filler_gap_dependency = index_of_word - index_of_head_noun
        relativizer_dict.update({"filler_gap_dependency": filler_gap_dependency})

        relativizer_count = 0
        # search the relativizer in the text from the head noun to the clause verb
        for index_of_relativizer in range(index_of_head_noun, index_of_clause_verb):
            # The relativizer need to meet two requirements:
            # a. match relativizer list
            # b. connecting directly with clausal root

            relativizer = dataset.words[i][index_of_relativizer]

            # nominal RC ( e.g. I eat what I like. )
            # the head noun is the relativizer
            if index_of_relativizer == dataset.arcs[i][index_of_clause_verb] - 1 and \
                    relativizer in ["what", "whatsoever", "whatever"]:
                if (relativizer in self.relativizer_list) and dataset.rels[i][
                    index_of_relativizer] != 'det':
                    if dataset.rels[i][index_of_relativizer - 1] == 'punct':
                        relativizer_dict.update({"restrictiveness": 'nonrestrictive'})
                    else:
                        relativizer_dict.update({"restrictiveness": 'restrictive'})

                    rc = self.relative_clause(subtrees, clause_verb, relativizer, text)

                    relativizer_dict.update({"relativizer": relativizer,
                                             "head_noun_in_rc": dataset.rels[i][index_of_relativizer],
                                             "relative_clause": rc})
                    relativizer_count += 1

            # whose, of which (e.g. the book whose cover is red) RC
            # a. match relativizer list  b. modifying nouns (e.g. whose, of which) connecting with clausal root
            elif (dataset.arcs[i][dataset.arcs[i][index_of_relativizer] - 1] == index_of_clause_verb + 1):
                if (relativizer in self.relativizer_list):
                    if dataset.rels[i][index_of_relativizer - 1] == 'punct':
                        relativizer_dict.update({"restrictiveness": 'nonrestrictive'})
                    else:
                        relativizer_dict.update({"restrictiveness": 'restrictive'})
                    rc = self.relative_clause(subtrees, clause_verb, relativizer, text)

                    relativizer_dict.update({"relativizer": relativizer,
                                             "head_noun_in_rc": dataset.rels[i][index_of_relativizer],
                                             "relative_clause": rc})
                    relativizer_count += 1


            # other types of RC
            elif (dataset.arcs[i][index_of_relativizer] == index_of_clause_verb + 1):
                # restrictiveness of RC
                if (relativizer in self.relativizer_list) and dataset.rels[i][
                    index_of_relativizer] != 'det':
                    if dataset.rels[i][index_of_relativizer - 1] == 'punct':
                        relativizer_dict.update({"restrictiveness": 'nonrestrictive'})
                    else:
                        relativizer_dict.update({"restrictiveness": 'restrictive'})

                    rc = self.relative_clause(subtrees, clause_verb, relativizer, text)

                    relativizer_dict.update({"relativizer": relativizer,
                                             "head_noun_in_rc": dataset.rels[i][index_of_relativizer],
                                             "relative_clause": rc})

                    relativizer_count += 1

        # if there is no relativizer, the relativizer is zero.
        if relativizer_count == 0 and dataset.rels[i][index_of_relativizer] != 'det':

            for word in text[index_of_head_noun:index_of_clause_verb]:
                if word in self.relativizer_list:
                    return None
            rc = self.zero_relative_clause(subtrees, clause_verb, text, index_of_head_noun)

            relativizer_dict.update({"restrictiveness": 'restrictive',
                                     "relativizer": "zero",
                                     "head_noun_in_rc": "dobj",
                                     "relative_clause": rc, })
        if not rc:
            return None

        relativizer_dict.update({"relative_clause_length": len(rc.split(" "))})

        return relativizer_dict

    def extract_relative_clauses(self, relativizer_list=None,
                                 role_of_relativizer=None,
                                 animacy_of_relativizer=None,
                                 restrictiveness=None):
        """
        main function
        :param relativizer_list: list of relativizers, e.g., which, that, zero, who, whose
        :param role_of_relativizer: list of roles of relativizers, e.g., nsubj, dobj
        :param animacy_of_relativizer: list of animacy of relativizers, e.g., animate, inanimate
        :param restrictiveness: list of restrictiveness of relative clauses, e.g., restrictive, nonrestrictive

        :return:
        """
        relativizer_dict_list = []

        files = os.listdir(self.input_folder)
        for file in files:
            # read text files
            if file.endswith(r".txt"):
                file_in = open(self.input_folder + file, "r", encoding='utf-8')
                for line in file_in:
                    # split text into sentences
                    sents = self.sent_tokenizer.tokenize(line)

                    # process each sentence
                    for sent in sents:

                        # tokenization, dependency parsing, constituency parsing
                        text, dataset, dataset2 = self.parsing(sent)
                        for i in range(len(dataset2.trees)):
                            # acquire all subtrees with SBAR label in the constituency parsing result
                            subtrees = self.acquire_subtrees(dataset2.trees[i])

                            # iterate each word in the sentence to identify relativizers and relative clauses in the sentence
                            for index_of_word in range(len(text)):
                                relativizer_dict = {}
                                # if the dependency relation of the word is "rcmod", there is a relative clause in the sentence
                                # "rcmod" also represents "acl:relcl" in the UD treebank
                                if dataset.rels[i][index_of_word] == "rcmod":
                                    # processing the relative clauses
                                    relativizer_result = self.process_relative_clause(text, dataset, index_of_word,
                                                                                      subtrees, i)
                                    if relativizer_result is not None:
                                        relativizer_dict.update({"file": file})
                                        relativizer_dict.update({"sent": sent})
                                        relativizer_dict.update(relativizer_result)

                                if len(relativizer_dict) > 0:
                                    relativizer_dict_list.append(relativizer_dict)

        # save the result to a csv file
        df = pd.DataFrame(relativizer_dict_list)
        df.columns = ["file", "sent", "antecedent", "head_noun_in_main_clause", "animacy", "clause verb",
                      "filler_gap_distance", "restrictiveness", "relativizer", "head_noun_in_rc", "relative_clause",
                      "relative_clause_length"]

        # filter the result
        if relativizer_list:
            df = df[df["relativizer"].isin(relativizer_list)]
        if role_of_relativizer:
            df = df[df["head_noun_in_rc"].isin(role_of_relativizer)]
        if animacy_of_relativizer:
            df = df[df["animacy"].isin(animacy_of_relativizer)]
        if restrictiveness:
            df = df[df["restrictiveness"].isin(restrictiveness)]

        df.to_csv(self.output_folder + "/result.csv", index=False)
        return (df.head(), f"""{len(df)} relative clauses have been identified and extracted. 
        The result has been saved to {self.output_folder}result.csv""")


if __name__ == '__main__':
    # input files directory
    input_text_dir = "input_texts/"  # replace with your own input directory

    # output text directory
    file_out_dir = "result/"  # replace with your own output directory

    RelativeClause = RelativeClause(input_text_dir, file_out_dir)

    # extract all relative clauses
    # relative_clauses = RelativeClause.extract_relative_clauses()

    # # You can also extract specific relative clauses
    specific_relative_clauses = RelativeClause.extract_relative_clauses(relativizer_list=["which", "that", "zero"],
                                                                        role_of_relativizer=["nsubj", "nsubjpass",
                                                                                             "dobj"],
                                                                        animacy_of_relativizer=["inanimate"],
                                                                        restrictiveness=["restrictive"])
