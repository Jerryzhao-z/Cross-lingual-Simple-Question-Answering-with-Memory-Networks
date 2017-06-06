# Cross-lingual Simple Question Answering with Memory Networks

## Prepare dataset and knowledge base

```bash

python3 ./data_prepared/summarize_corpus.py

python3 ./data_prepared/summarize_facts.py

python3 ./data_prepared/divide_corpus.py

```

## Train MemNN

```
python3 SQA.py
```

# continue training MemNN with saved checkpoint

```
python3 SQA.py --from_checkpoint True
```

## Test MemNN

```
python3 SQA.py --state 1 --from_checkpoint True
```

## QA with trained MemNN

```
python3 SQA.py --state 2 --from_checkpoint True
```

-----------------

## folders and files

/checkpoint : 
    
    the default folder to save checkpoint of MemNN

/corpus :

    /original:

        the original datasets

    sum_corpus.txt:

         default file which collect all dataset in /original folder

    testset.txt: 

        default test dataset

    trainset.txt: 

        default train dataset

/data_prepared  :

    divide_corpus.py:

        divide the /corpus/sum_corpus.txt to /corpus/trainset.txt and /corpus/testset.txt

    summarize_corpus.py:

        summarize dataset in /corpus/original in /corpus/sum_corpus.txt

    summarize_fact.py:

        summarize knowledge base in /facts/original in /corpus/sum_facts.txt

/facts :

    /original:

        the original knowledge base

    sum_facts.txt:

        default file which collect all knowledge base in /original folder

/intermediate:

    save intermediate files generated in training process and used in test/answer process

/MemNN:

    MemNN and its components

/transform:

    json object which could map mid/IMDBid to labels and map labels to mid/IMDBid 

common.py

SQA.py:

    main file of the project