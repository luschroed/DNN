import torch


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    save_best_model=False,
    model_path="results/best_model.pt"
):

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_accuracy = 0.0
    best_epoch = 0
    best_state_dict = None

    for epoch in range(epochs):

        # ==================================================
        # Training
        # ==================================================

        model.train()

        running_train_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_train_loss += loss.item()

        train_loss = (
            running_train_loss / len(train_loader)
        )

        # ==================================================
        # Validation
        # ==================================================

        model.eval()

        running_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                running_val_loss += loss.item()

                _, predicted = torch.max(
                    outputs,
                    1
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        val_loss = (
            running_val_loss / len(val_loader)
        )

        val_accuracy = (
            100 * correct / total
        )

        # ==================================================
        # Save history
        # ==================================================

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # ==================================================
        # Track best model
        # ==================================================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            if save_best_model:

                torch.save(
                    model.state_dict(),
                    model_path
                )

        # ==================================================
        # Progress
        # ==================================================

        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

    # ======================================================
    # Restore best model
    # ======================================================

    if best_state_dict is not None:

        model.load_state_dict(
            best_state_dict
        )

    history["best_val_accuracy"] = best_val_accuracy
    history["best_epoch"] = best_epoch

    return history
