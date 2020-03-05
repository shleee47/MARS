import csv

# label, youtube_id, time_start, time_end, split, is_cc 
def categories(fin):
    list = []
    with open(fin, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['label'] in list:
                continue
            else:
                list.append(row['label'])

    return list

def list2txt(list, fout):
    with open(fout, 'w') as f:
        for item in list:
            f.write("%s" % item)
            f.write("\n")

def txt2list(fin):
    """
    """

def manipulate(fin, fout, categories):
#    with open(fin, newline='') as csvfile:
#        reader = csv.DictReader(csvfile)
#        for row in reader:
#            print(row['label'])
    fieldnames = ['label', 'youtube_id', 'time_start', 'time_end', 'split', 'is_cc']
#    kept = []
    kept = {}
    with open(fin, 'r') as csvfile, open(fout, 'w') as outputfile:
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        writer = csv.DictWriter(outputfile, fieldnames=fieldnames)
        for row in reader:
            # if label == related categories, keep it
            if row['label'] in categories:
                writer.writerow({'label': row['label'], 
                        'youtube_id': row['youtube_id'], 
                        'time_start': row['time_start'],
                        'time_end': row['time_end'],
                        'split': row['split'],
                        'is_cc': row['is_cc']})

            # if label != related categories, keep only one data
            else:
                if row['label'] in kept:
                    if kept[row['label']] == 2:
                        continue
                    else:
                        writer.writerow({'label': row['label'], 
                                'youtube_id': row['youtube_id'], 
                                'time_start': row['time_start'],
                                'time_end': row['time_end'],
                                'split': row['split'],
                                'is_cc': row['is_cc']})
                        kept[row['label']] += 1

                else:
                    writer.writerow({'label': row['label'], 
                            'youtube_id': row['youtube_id'], 
                            'time_start': row['time_start'],
                            'time_end': row['time_end'],
                            'split': row['split'],
                            'is_cc': row['is_cc']})
                    kept[row['label']] = 1

#                # if already have, pass rest
#                if row['label'] in kept:
#                    continue
#                # if don't have, keep it
#                else:
#                    writer.writerow({'label': row['label'], 
#                            'youtube_id': row['youtube_id'], 
#                            'time_start': row['time_start'],
#                            'time_end': row['time_end'],
#                            'split': row['split'],
#                            'is_cc': row['is_cc']})
#                    kept.append(row['label'])

                


if __name__ == '__main__':
    categories = ['capoeira',
           'drop kickinga',
           'faceplanting',
           'headbutting',
           'high kick',
           'playing ice hockey',
           'punching bag',
           'punching person (boxing)',
           'side kick',
           'slapping',
           'sword fighting',
           'tai chi',
           'wrestling']
    manipulate('kinetics-400_val.csv', 'kinetics-400_val-light.csv', categories)

#    list2txt(list, 'categories.txt')
#    manipulate('kinetics-400_val.csv', 'aaa')
