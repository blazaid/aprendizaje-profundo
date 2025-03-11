import torch
import time


def train(
    model,
    train_loader, 
    n_epochs,
    criterion,
    optimizer,
    validation_split=None,
    metric_fn=None,
    verbose=True,
):
    history = {"train_loss": []}

    if metric_fn is not None:
        history["train_metric"] = []

    # Si se especifica validation_split, dividimos el conjunto de entrenamiento
    if validation_split is not None and 0 < validation_split < 1:
        total_size = len(train_loader.dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(train_loader.dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_loader.batch_size, shuffle=False, num_workers=2)

        history["val_loss"] = []
        if metric_fn is not None:
            history["val_metric"] = []
    else:
        val_loader = None

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        if metric_fn is not None:
            metric_fn.reset()

        start_time = time.time()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            if metric_fn is not None:
                metric_fn.update(outputs, labels)
        elapsed = time.time() - start_time

        avg_train_loss = running_loss / total_samples
        history["train_loss"].append(avg_train_loss)

        msg = f"Epoch {epoch + 1}/{n_epochs} ({elapsed:.2f}s), Train loss: {avg_train_loss:.4f}"

        if metric_fn is not None:
            avg_train_metric = metric_fn.compute().item()
            history["train_metric"].append(avg_train_metric)
            msg += f", Train metric: {avg_train_metric:.4f}"

        if val_loader:
            model.eval()
            val_loss = 0.0
            val_samples = 0
            if metric_fn is not None:
                metric_fn.reset()

            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    batch_size = images.size(0)
                    val_loss += loss.item() * batch_size
                    val_samples += batch_size

                    if metric_fn is not None:
                        metric_fn.update(outputs, labels)

            avg_val_loss = val_loss / val_samples
            history["val_loss"].append(avg_val_loss)
            msg += f", Val loss: {avg_val_loss:.4f}"

            if metric_fn is not None:
                avg_val_metric = metric_fn.compute().item()
                history["val_metric"].append(avg_val_metric)
                msg += f", Val. metric: {avg_val_metric:.4f}"

        if verbose:
            print(msg)

    return history


def evaluate(*, model, data_loader, criterion, metric_fn=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_value = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            if metric_fn is not None:
                metric_value += metric_fn(outputs, labels) * images.size(0)

    avg_loss = total_loss / total_samples
    avg_metr = (metric_value / total_samples) if metric_fn is not None else None

    return {"loss": avg_loss, "metric": avg_metr}


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

device = get_device()
print(f'Using device: {device}')
