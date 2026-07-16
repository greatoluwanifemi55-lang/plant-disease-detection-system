from plant_fl.flower_client import FlowerClient
from plant_fl.federated_server import FederatedServer
from plant_fl.metrics import (
    initialize_metrics,
    save_metrics,
)

# =====================================================
# CONFIGURATION
# =====================================================

NUM_ROUNDS = 10
LOCAL_EPOCHS = 2


# =====================================================
# HELPER FUNCTION
# =====================================================

def separator():
    print("=" * 70)


# =====================================================
# MAIN SIMULATION
# =====================================================

def main():

    separator()
    print("PLANT DISEASE FEDERATED LEARNING SIMULATION")
    separator()

    print("\nLoading clients...\n")

    clients = [
        FlowerClient("Oyo"),
        FlowerClient("Kaduna"),
        FlowerClient("Benue"),
    ]

    print("\nAll clients loaded successfully!")

    server = FederatedServer()

    # Create metrics.csv
    initialize_metrics()

    # Keep track of the best model
    best_accuracy = 0.0

    # Initial global weights
    global_weights = clients[0].get_weights()

    # =====================================================
    # FEDERATED TRAINING
    # =====================================================

    for rnd in range(NUM_ROUNDS):

        separator()
        print(f"FEDERATED ROUND {rnd + 1} OF {NUM_ROUNDS}")
        separator()

        print("\nSending global model to all clients...")

        for client in clients:
            client.set_weights(global_weights)

        print("\nStarting local training...\n")

        for client in clients:

            print(f"Training client: {client.client_name}")

            client.train(
                epochs=LOCAL_EPOCHS
            )

            print(f"{client.client_name} finished training.\n")

        print("Aggregating client models using Weighted FedAvg...")

        global_weights = server.aggregate(clients)

        print("Aggregation complete.\n")

        # Update clients with new global model
        for client in clients:
            client.set_weights(global_weights)

        separator()
        print(f"EVALUATION AFTER ROUND {rnd + 1}")
        separator()

        accuracies = []

        for client in clients:

            loss, accuracy = client.evaluate()

            accuracies.append(accuracy)

            save_metrics(
                rnd + 1,
                client.client_name,
                loss,
                accuracy,
            )

            print(
                f"{client.client_name:10} | "
                f"Loss: {loss:.4f} | "
                f"Accuracy: {accuracy:.4f}"
            )

        average_accuracy = sum(accuracies) / len(accuracies)

        print(f"\nGlobal Average Accuracy: {average_accuracy:.4f}")

        # Save the best model
        if average_accuracy > best_accuracy:

            best_accuracy = average_accuracy

            clients[0].model.set_weights(global_weights)

            clients[0].model.save(
                "experiment_results/best_global_model.keras"
            )

            print(
                f"\nNew Best Model Saved! "
                f"(Accuracy = {best_accuracy:.4f})"
            )

    # =====================================================
    # SAVE FINAL MODEL
    # =====================================================

    separator()
    print("Saving Final Global Model...")
    separator()

    clients[0].model.set_weights(global_weights)

    clients[0].model.save("global_model.keras")

    print("Final global model saved successfully!")

    # =====================================================
    # SAVE EXPERIMENT SUMMARY
    # =====================================================

    with open(
        "experiment_results/experiment_summary.txt",
        "w"
    ) as file:

        file.write("Plant Disease Federated Learning Experiment\n")
        file.write("=" * 45 + "\n\n")

        file.write(f"Federated Rounds : {NUM_ROUNDS}\n")
        file.write(f"Local Epochs     : {LOCAL_EPOCHS}\n")
        file.write(f"Best Accuracy    : {best_accuracy:.4f}\n")

    separator()
    print("FEDERATED TRAINING COMPLETED")
    separator()

    print("\nGenerated Files:")

    print("✔ experiment_results/metrics.csv")
    print("✔ experiment_results/best_global_model.keras")
    print("✔ experiment_results/experiment_summary.txt")
    print("✔ global_model.keras")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()