import torch
import torch.optim as optim
from gan_model import Generator, Discriminator
from data_utils import load_fmri_data, preprocess_data, save_generated_data
import os
import numpy as np

# Hyperparameters
latent_dim = 100
data_dim =  # Replace with actual dimension of your fMRI data
learning_rate = 0.0002
batch_size = 64
num_epochs = 100

# Load and preprocess data
data_path = 'data/fmri_data.npy'
real_data = load_fmri_data(data_path)
real_data = preprocess_data(real_data)
real_data = torch.tensor(real_data, dtype=torch.float32)

# Initialize models and optimizers
generator = Generator(latent_dim, data_dim)
discriminator = Discriminator(data_dim)
generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(real_data), batch_size):
        real_batch = real_data[i:i+batch_size]
        # Train Discriminator
        discriminator_optimizer.zero_grad()
        real_labels = torch.ones(real_batch.size(0), 1)
        fake_labels = torch.zeros(real_batch.size(0), 1)

        real_output = discriminator(real_batch)
        d_loss_real = criterion(real_output, real_labels)

        z = torch.randn(real_batch.size(0), latent_dim)
        fake_data = generator(z)
        fake_output = discriminator(fake_data)
        d_loss_fake = criterion(fake_output, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        discriminator_optimizer.step()

        # Train Generator
        generator_optimizer.zero_grad()
        z = torch.randn(real_batch.size(0), latent_dim)
        fake_data = generator(z)
        output = discriminator(fake_data)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        generator_optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')

    # Save model checkpoints periodically
    if (epoch + 1) % 10 == 0:
        os.makedirs('results/checkpoints', exist_ok=True)
        torch.save(generator.state_dict(), f'results/checkpoints/generator_{epoch+1}.pth')
        torch.save(discriminator.state_dict(), f'results/checkpoints/discriminator_{epoch+1}.pth')

# Generate and save synthetic data after training
num_samples = 1000
z = torch.randn(num_samples, latent_dim)
generated_data = generator(z).detach().numpy()
save_generated_data(generated_data, 'results/generated_fmri_data.npy')
