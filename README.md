For TASK 2:

dataset.py create .npy data from the orginal data, and create dataloader for pytorch model.

model.py create two models, one is MLP, the other is modified MLP with Fourier neural operator. Modified MLP makes use of the advantages of Fourier neural operator to learn the pressure field and will have the more powerful generalization ability.

main.py load the data from dataloader and split it into train, validation and test dataloader, with 4:2:4. The train loss is pytorch l1loss(MAEloss). Then it will train the model and test the model in MAE.

[data link](https://drive.google.com/drive/folders/1Ynb4EbkFYsGdpG82T6cjJLr0Kht3Cp9V?usp=drive_link)
