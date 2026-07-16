import tensorflow as tf

from plant_fl.task import (
    load_model,
    load_data,
    train_model,
    evaluate_model,
)


class FlowerClient:

    def __init__(self, client_name):

        self.client_name = client_name

        print(f"\nLoading dataset for {client_name}...")

        (
        self.train_dataset,
        self.val_dataset,
        self.class_names,
        self.num_examples,
        ) = load_data(client_name)

        self.model = load_model()

        

    def get_weights(self):

        return self.model.get_weights()

    def set_weights(self, weights):

        self.model.set_weights(weights)

    def train(self, epochs=1):

        print(f"\nTraining {self.client_name}...")

        history = train_model(
            self.model,
            self.train_dataset,
            epochs,
        )

        return history.history

    def evaluate(self):

        loss, accuracy = evaluate_model(
            self.model,
            self.val_dataset,
        )

        return loss, accuracy