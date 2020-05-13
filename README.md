# Birds Eye View Prediction

Running the scripts from the final_integration directory:
1. Run train_autoencoder.sh script to train an autoencoder model
2. Run train_rm.sh script to train the six ResNets for Road Map Generation Task
3. Run train_rotation.sh script to pretrain a ResNet model using SSL rotation pretext task
4. Run generate_mono_data.sh script to generate training data for predicting masks for dynamic elements
5. Run train_bb.sh script to train the GAN for predicting masks using the generated data