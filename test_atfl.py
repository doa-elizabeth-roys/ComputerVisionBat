import torch
import torch.nn as nn

# Simulate predictions (logits) and ground-truth labels
torch.manual_seed(0)
pred = torch.randn((4, 1), requires_grad=True)  # raw logits
true = torch.randint(0, 2, (4, 1)).float()       # binary targets

# BCE loss function
loss_fcn = nn.BCEWithLogitsLoss(reduction='none')

# Step 1: Compute basic BCE loss
loss = loss_fcn(pred, true)

# Step 2: Get predicted probabilities
pred_prob = torch.sigmoid(pred)

# Step 3: Compute p_t (confidence of correct class)
p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
mean_pt = p_t.mean()

# Step 4: Update p_t moving average (simulate only one value here)
p_t_old = mean_pt
p_t_new = 0.05 * p_t_old + 0.95 * mean_pt

# Step 5: Compute adaptive gamma
gamma = -torch.log(p_t_new)

# Step 6: Modulate loss
p_t_high = torch.where(p_t > 0.5, (1.000001 - p_t)**gamma, torch.zeros_like(p_t))
p_t_low = torch.where(p_t <= 0.5, (1.5 - p_t)**(-torch.log(p_t)), torch.zeros_like(p_t))
modulating_factor = p_t_high + p_t_low

# Step 7: Apply modulation
adjusted_loss = loss * modulating_factor

# Step 8: Print everything
print("Pred (logits):", pred.view(-1).detach().numpy())
print("True labels:", true.view(-1).numpy())
print("Predicted probabilities:", pred_prob.view(-1).detach().numpy())
print("p_t:", p_t.view(-1).detach().numpy())
print("mean p_t:", mean_pt.item())
print("gamma:", gamma.item())
print("modulating factor:", modulating_factor.view(-1).detach().numpy())
print("Original loss:", loss.view(-1).detach().numpy())
print("Adjusted loss:", adjusted_loss.view(-1).detach().numpy())

from ultralytics.utils.loss import AdaptiveThresholdFocalLoss


# Create dummy predictions and targets
pred = torch.randn(5, requires_grad=True)
target = torch.randint(0, 2, (5,), dtype=torch.float32)

# Instantiate your loss
loss_fn = AdaptiveThresholdFocalLoss(torch.nn.BCEWithLogitsLoss(reduction='none'))

# Calculate loss
loss = loss_fn(pred, target)
print("Loss:", loss.mean().item())

if torch.rand(1).item() < 0.01:  # print occasionally
    print(f"mean_pt: {mean_pt.item():.4f}, p_t_old: {self.p_t_old.item():.4f}, gamma: {gamma.item():.4f}")

from ultralytics.utils.loss import AdaptiveThresholdFocalLoss
import torch
import torch.nn as nn

# Create dummy predictions and targets
pred = torch.randn(5, requires_grad=True)
target = torch.randint(0, 2, (5,), dtype=torch.float32)

# Instantiate your loss
loss_fn = AdaptiveThresholdFocalLoss(nn.BCEWithLogitsLoss(reduction='none'))

# Calculate loss
loss = loss_fn(pred, target)
print("Using ATFL loss!")
print("Loss:", loss.mean().item())  # print mean to avoid tensor size error
