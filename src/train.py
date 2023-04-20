import torch

def eval(transformer, loss_fn, device, dl):
    transformer.eval()
    with torch.no_grad():
        total_loss = 0
        total_acc = 0
        for _, data in enumerate(dl):
            x, y = data[0].to(device), data[1].to(device)
            output = transformer(x)
            
            # Flatten batch and sequence dimension
            loss = loss_fn(output.view(-1, output.shape[-1]), y.view(-1)).item()
            total_loss += loss
            pred = torch.nn.functional.softmax(output, dim=-1)
            pred = pred.argmax(dim=-1)
            accuracy = (pred == y).float().mean().item()
            total_acc += accuracy
    return total_loss / len(dl), total_acc / len(dl)


def train(transformer, loss_fn, optim, device, dl):
    transformer.train()
    for _, data in enumerate(dl):
        x, y = data[0].to(device), data[1].to(device)
        output = transformer(x)
        
        # Flatten batch and sequence dimension
        loss = loss_fn(output.view(-1, output.shape[-1]), y.view(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()