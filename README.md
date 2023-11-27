# ssim

# We used pre-trained weight from here: https://github.com/google-research/big_transfer
#### Folder
```
root/    
|-- data/
    |-- ISIC/
    |   |-- TrainDataset/
    |   |   |-- images/
    |   |   |   |-- ISIC_0000001.jpg 
    |   |   |   |-- ISIC_0000002.jpg 
    |   |   |   ...
    |   |   |-- masks/
    |   |       |-- ISIC_0000001.jpg 
    |   |       |-- ISIC_0000002.jpg 
    |   |       ...
    |   |-- ValidationDataset/
    |   |   |-- images/
    |   |   |   |-- ISIC_0000003.jpg 
    |   |   |   |-- ISIC_0000004.jpg 
    |   |   |   ...
    |   |   |-- masks/
    |   |       |-- ISIC_0000003.jpg 
    |   |       |-- ISIC_0000004.jpg 
    |   |       ...
    |   |-- TestDataset/
    |       |-- images/
    |       |   |-- ISIC_0000005.jpg 
    |       |   |-- ISIC_0000006.jpg 
    |       |   ...
    |       |-- masks/
    |           |-- ISIC_0000005.jpg 
    |           |-- ISIC_0000006.jpg 
    |           ...
    |-- Kvasir-SEG/
        |-- TrainDataset/
        |   |-- images/
        |   |   ...
        |   |-- masks/
        |       ...
        |-- ValidationDataset/
        |   |-- images/
        |   |   ...
        |   |-- masks/
        |       ...
        |-- TestDataset/
            |-- images/
            |   ...
            |-- masks/
                ...
```

####  2. Training

```bash
python main.py --config ./configuration/isic.yaml
```

####  3. Testing

```bash
python main.py --config ./configuration/isic.yaml --test
```
