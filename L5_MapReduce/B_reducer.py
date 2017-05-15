import sys


def reducer():
    word_count = 0
    old_key = None

    for line in sys.stdin:
        data = line.strip().split("\t")

        if len(data) != 2:
            continue

        key, count = data

        if old_key and (old_key != key):
            print('{0}\t{1}'.format(old_key, word_count))
            word_count = 0

        old_key = key
        word_count += int(count)

    if old_key is not None:
        print('{0}\t{1}'.format(old_key, word_count))


reducer()

#  $ cat Alice.txt | python B_mapper.py | sort | python B_reducer.py
