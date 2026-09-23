# CICIDS2017 Dataset

## Citation

Iman Sharafaldin, Arash Habibi Lashkari, and Ali A.
Ghorbani, “[Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization](https://www.scitepress.org/PublicationsDetail.aspx?ID=K3WXGO8/3O4=&t=1)”,
4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018.

## Dataset Description

Due to dataset size, the CICIDS2017 dataset is not included in this repository. However, you can download it from the
official website of the Canadian Institute for Cybersecurity (CIC) at the following link:
<https://www.unb.ca/cic/datasets/ids-2017.html>.

The dataset contains various files, including CSV files for different days of network traffic, which can be used for
training and evaluating intrusion detection models. Make sure to follow any instructions provided by the CIC for proper
usage of the dataset.

Once you have downloaded the dataset, move the CSV files located in the `MachineLearningCSV.zip` folder to the
`{PROJECT_ROOT}/datasets/cicids2017/` directory in this repository. This will allow you to access the dataset for your
machine learning tasks.

The CSV files we want to use for training and evaluation are:

- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
- Friday-WorkingHours-Morning.pcap_ISCX.csv
- Monday-WorkingHours.pcap_ISCX.csv
- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
- Tuesday-WorkingHours.pcap_ISCX.csv
- Wednesday-workingHours.pcap_ISCX.csv
