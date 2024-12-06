## Part 0 - Comment Request from Ethan

0. The objective for the changes:
    - We want to time march the model prediction process so that it can reuse previous predictions to make future predictions. e.g.,
        a. Train x1, x2, x3, x4, x5 -> x6
        b. Train x2, x3, x4, x5, x6* -> x7
        c. Train x3, x4, x5, x6*, x7* -> x8
        d. Train x4, x5, x6*, x7*, x8* -> x9
        e. Train x5, x6*, x7*, x8*, x9* -> x10

    Here x6*, x7*, x8*, x9* are the past predictions made by the model. The model should be able to reuse these predictions to make future predictions.

1. Review Required on two files:
    ```
    ./model_training/models/deeponet.py
    ./model_training/data/dataset_fpo.py
    ```

2. Changes Made in deeponet.py
    - Added method ```on_train_epoch_end()```. This allows us to perform the following operations at the end of each epoch:
        - Save the current prediction to a ```.npy``` file
        - Reload the Dataset object by discarding the first prediction and adding the new prediction
        - Do it every 100 epochs
        - Calls an ```update_data()``` method in the ```DataModule``` object to update the dataset

3. Changes Made in dataset_fpo.py
    - Added class ```FPODataModule``` and in it a method ```update_data()```. This allows us to perform the following operations:
        - Load the current prediction from the ```.npy``` file
        - Performs the full range of time stepping calculations to update the dataset
        - Uses the time calculations to call the ```DataLoader``` object to update the dataset.

    - Added a new ```FPODatasetMix``` dataloader that allows us to mix the current prediction with the actual data to train the model, given a set of time indices.

## Part - I : Data Preprocessing - Status : Completed, Tested. Done

0. Request an Interactive Node and activate the virtual environment on Nova:
    ```
    salloc -N 2 -n 72 -t 08:00:00 --mem 369G
    source /work/mech-ai/rtali/mlpythonenv/bin/activate
    ```

1. Check the data settings here:
    ```
    ./dataset_generation/data.ini
    ```
2. Run the following command to generate the dataset:
    ```
    python3 ./dataset_generation/data_prep.py
    ```

3. Two numpy tensor files will be generated:
    ```
    ./dataset_generation/fpo/in_data.npz
    ./dataset_generation/fpo/out_data.npz
    ```

## Part - II : Model Training - Status : Initial Code Changes Done. Finalize after Code Review

0. Request an Interactive GPU Node and activate the virtual environment on Nova:
    ```
    salloc -N 1 -n 8 --gres gpu:a100:1 -t 8:00:00 --mem 369G
    source /work/mech-ai/rtali/mlpythonenv/bin/activate
    ```