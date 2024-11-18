## Part - I : Data Preprocessing

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

## Part - II : Model Training

0. Request an Interactive GPU Node and activate the virtual environment on Nova:
    ```
    salloc -N 1 -n 8 --gres gpu:a100:1 -t 8:00:00 --mem 369G
    source /work/mech-ai/rtali/mlpythonenv/bin/activate
    ```