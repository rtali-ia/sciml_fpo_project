##Part - I : Data Preprocessing

0. Activate the virtual environment on Nova:
    ```
    source /work/mech-ai/rtali/mlpythonenv/bin/activate
    ```

1. Check the data settings here:
    ```
    /dataset_generation/data.ini
    ```
2. Run the following command to generate the dataset:
    ```
    python3 /dataset_generation/data_prep.py
    ```

3. Two numpy tensor files will be generated:
    ```
    /dataset_generation/fpo/in_data.npz
    /dataset_generation/fpo/out_data.npz
    ```

##Part - II : Model Training