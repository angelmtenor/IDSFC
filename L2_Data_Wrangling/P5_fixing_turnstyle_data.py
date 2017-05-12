import csv
import os

DIR = 'MTA_Subway_turnstile'


def fix_turnstile_data(filenames):
    """
    Filenames is a list of MTA Subway turnstile text files. A link to an example
    MTA Subway turnstile text file can be seen at the URL below:
    http://web.mta.info/developers/data/nyct/turnstile/turnstile_110507.txt

    As you can see, there are numerous data points included in each row of the
    a MTA Subway turnstile text file. 

    You want to write a function that will update each row in the text
    file so there is only one entry per row. A few examples below:
    A002,R051,02-00-00,05-28-11,00:00:00,REGULAR,003178521,001100739
    A002,R051,02-00-00,05-28-11,04:00:00,REGULAR,003178541,001100746
    A002,R051,02-00-00,05-28-11,08:00:00,REGULAR,003178559,001100775

    Write the updates to a different text file in the format of "updated_" + filename.
    For example:
        1) if you read in a text file called "turnstile_110521.txt"
        2) you should write the updated data to "updated_turnstile_110521.txt"

    The order of the fields should be preserved. Remember to read through the 
    Instructor Notes below for more details on the task. 

    In addition, here is a CSV reader/writer introductory tutorial:
    http://goo.gl/HBbvyy

    You can see a sample of the turnstile text file that's passed into this function
    and the the corresponding updated file by downloading these files from the resources:

    Sample input file: turnstile_110528.txt
    Sample updated file: solution_turnstile_110528.txt
    """
    for name in filenames:
        # your code here
        f_in = open(DIR + '/' + name, 'r')
        f_out = open(DIR + '/' + 'updated_' + name, 'w')

        reader_in = csv.reader(f_in, delimiter=',')
        writer_out = csv.writer(f_out, delimiter=',')

        for line in reader_in:
            fixed_fields = line[0:3]
            for i in range(3, len(line), 5):
                row = fixed_fields + line[i:i + 5]
                writer_out.writerow(row)
        f_in.close()
        f_out.close()

        print(name, ' fixed')
    return


turnstile_filenames = [f for f in os.listdir(DIR) if f.startswith("turnstile")]

fix_turnstile_data(turnstile_filenames)
