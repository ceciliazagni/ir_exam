# Information retrieval project
The project contained in this repository is project number 5: UPDATE THE DICTIONARY.

## Content of this repository
- `interactive_update_the_dictionary_cz.py`: python3 file containing the project
- `myIRsystem.sh`: bash file used to launch the creation of the IR systems in an `ORFEO`'s `EPYC` node
- `movie.metadata.tsv`: file used to create the corpus
- `plot_summaries.txt`: file used to create the corpus
- `/ready_to_use`: contains some IR systems that can be uploaded to perform some test queries. (sent via mail)

## How to create an IR system
In order to create an IR system starting from the "mvies dataset" (movie.metadata.tsv + plot_summaries.txt) do the following (after you have loaded your pyton3 environment):

```
python3 interactive_update_the_dictionary_cz.py
```

The corpus is divided in 3 parts:
- A contains corpus[:25000]
- B contains corpus[25000:36000]
- C contains corpus[36000:]

The following files are generated:

- `my_ir_AB.pkl` is built indexing A and B
- `my_ir_ABC.pkl` is built adding C to the previous IR system
- `my_ir_AC.pkl` is built removing B from the previous IR system
- `my_ir_AC_fast_merge.pkl` is built performing a "fast merge" to the previous IR system

A version of these files is present in the `/ready_to_use` folder.

## How to load and test an IR system
In order to load and test with some queries an already existing IR system, do the following (after you have loaded your pyton3 environment):

```
python3 interactive_update_the_dictionary_cz.py upload
```

The program will display the following message:

```
Enter the exact filename of the IR system to upload:
```

For example, insert `/ready_to_use/my_ir_ABC.pkl`.

The program will display the following message:

```
The loaded file is of type: ...
Some info about the loaded file:
...
-------------------------------------------------------------

Rules for valid queries:
- Prase queries must be enclosed in ""
- If you need to search one single word, enclose it in "" (like a phrase query)
- Boolean queries must contain at least one boolean operator (AND,OR, NOT) and may contain parenthesis. They must not contain "

Enter a valid query or press Enter to exit:

```


### Some example of valid queries
- `"hulk"`
- `yoda OR kenobi`
- `(yoda OR kenobi) AND NOT leila`
- `"no place like home"`
- `"hakuna matata"`
