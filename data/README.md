# Dataset

Client order data is not included in this repository. Obtain the approved privacy-reduced `Modalii_Lenshub_ML_Dataset_Final.csv` from the group.

For local notebooks, place the CSV in this folder. For the command-line workflow, pass its path with `--dataset`. The CSV is ignored by Git. Do not force-add client data without permission to distribute it.

The expected stored fields are `Order_Record_ID`, `Delivery Charge`, `Weight`, `COD`, `Delivery_Charge_Ratio`, `Is_Outstation` and `Is_Returned`. The code creates `COD_Is_Zero` and recomputes the ratio. See `data_dictionary.csv` for definitions.
