from plant_fl.flower_client import FlowerClient

print("=" * 50)
print("Testing Oyo Client")
print("=" * 50)

client = FlowerClient("Oyo")

print(f"\nNumber of classes: {len(client.class_names)}")
print(client.class_names)

history = client.train(epochs=1)

loss, accuracy = client.evaluate()

print("\n==============================")
print("Evaluation Results")
print("==============================")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")