import string
import sys


def mapper():

    for line in sys.stdin:
        data = line.strip().split(" ")

        for i in data:
            cleaned_data = i.translate(str.maketrans('', '', string.punctuation)).lower()
            # emit a key-value pair
            print("{0}\t{1}".format(cleaned_data, 1))

mapper()
