### Description of scripts in `model` directory

- `add_jga_new_conditional.py`: (rarely used) post-hoc addition of conditional JGA scores using the `report` files and `world_log` files.
- `rename_folder_with_opts.py`: unify naming structure using `model.opt` files in each directory
- `show_test_results.py`: print out jga results. 
- `summarize_jga.ipynb`: takes code in `show_test_results.py` to summarize jga more interactively 
    - also written to accommodate faulty json code in case `.trainstats` code is problematic