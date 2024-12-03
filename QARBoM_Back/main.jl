using QARBoM

rbm = QARBoM.RBM(3,20)

# Train the RBM using Persistent Contrastive Divergence
N_EPOCHS = 100
BATCH_SIZE = 10

train_data = [[1, 0, 0] for _ in 1:1000]

QARBoM.train!(
    rbm, 
    train_data,
    CD; 
    n_epochs = N_EPOCHS,  
    cd_steps = 3, # number of gibbs sampling steps
    learning_rate = [0.0001/(j^0.8) for j in 1:N_EPOCHS], 
    metrics = [MeanSquaredError], # the metrics you want to track
    early_stopping = true,
    file_path = "my_cd_metrics.csv",
)

