from training.utils.train_loop import train_loop


def run_n_epochs(epochs, dataloader, model, loss_fn, optimizer, idx2char):
    for t in range(epochs):
        print(f"Epoch {t}\n-------------------------------")
        train_loop(dataloader, model, loss_fn, optimizer, t, idx2char)
